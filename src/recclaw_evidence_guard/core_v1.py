"""Development-only evidence guard for RecClaw exploration clients.

This module is intentionally not an authority plane.  It helps a Candidate
Producer or Router detect protocol-changing proposals and classify historical
observations before they enter search memory.  It never issues an
ActionContract, EvidenceAdmissionDecision, AdmissibleEvent, claim transition,
permission, promotion, or gate decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


PROFILE_ACTION_FAMILY = {
    "OFFLINE_TOPN": "RUN_OFFLINE_TOPN",
    "SEQUENTIAL_NEXT_ITEM": "RUN_SEQUENTIAL_NEXT_ITEM",
}
PROTOCOL_FIELDS = {
    "protocol_id",
    "profile_family",
    "dataset",
    "dataset_snapshot",
    "split",
    "training_sampling",
    "evaluation_candidate_universe",
    "candidate_policy",
    "metric",
    "training_procedure",
}
CLAIM_FIELDS = {
    "claim_id",
    "protocol_id",
    "claim_kind",
    "target_model",
    "comparator",
    "metric",
    "required_seed_count",
    "scope",
}
EVIDENCE_FIELDS = {"snapshot_id", "claim_id", "protocol_id", "observation_ids"}
PROPOSAL_FIELDS = {
    "proposal_id",
    "action_family",
    "target_claim_id",
    "protocol_id",
    "planned_protocol",
    "target_model",
    "comparator",
    "seed_count",
    "seed_ids",
    "purpose",
}
OBSERVATION_FIELDS = {
    "observation_id",
    "proposal_id",
    "claim_id",
    "protocol_id",
    "observed_protocol",
    "target_model",
    "comparator",
    "seed_count",
    "seed_runs",
    "observation_kind",
    "run_status",
    "artifact_identity_status",
    "evidence_class",
    "metrics",
    "notes",
}


class GuardInputError(ValueError):
    def __init__(self, code: str, message: str, *, path: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


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
        raise GuardInputError(
            "GUARD_NONCANONICAL_INPUT",
            "input must contain only finite JSON values",
            path="/",
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _object(
    value: object, fields: set[str], *, label: str, path: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GuardInputError(
            "GUARD_INPUT_TYPE",
            f"{label} must be an object",
            path=path,
        )
    result = {str(key): deepcopy(item) for key, item in value.items()}
    if set(result) != fields:
        raise GuardInputError(
            "GUARD_INPUT_FIELDS",
            f"{label} fields are not closed; missing={sorted(fields - set(result))} "
            f"extra={sorted(set(result) - fields)}",
            path=path,
        )
    _json_bytes(result)
    return result


def _string(value: object, *, label: str, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise GuardInputError(
            "GUARD_INPUT_VALUE", f"{label} must be a non-empty string", path=path
        )
    return value


def _positive_int(value: object, *, label: str, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GuardInputError(
            "GUARD_INPUT_VALUE", f"{label} must be a positive integer", path=path
        )
    return value


def _sha256(value: object, *, label: str, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise GuardInputError(
            "GUARD_INPUT_VALUE", f"{label} must be a lowercase SHA-256", path=path
        )
    return value


def _validate_protocol(value: object, *, path: str) -> dict[str, Any]:
    protocol = _object(value, PROTOCOL_FIELDS, label="Protocol", path=path)
    for field in ("protocol_id", "profile_family", "dataset", "dataset_snapshot"):
        _string(protocol[field], label=f"Protocol.{field}", path=f"{path}/{field}")
    if protocol["profile_family"] not in PROFILE_ACTION_FAMILY:
        raise GuardInputError(
            "GUARD_UNSUPPORTED_PROFILE",
            "Evidence Guard supports only OFFLINE_TOPN and SEQUENTIAL_NEXT_ITEM",
            path=f"{path}/profile_family",
        )
    for field in (
        "split",
        "training_sampling",
        "evaluation_candidate_universe",
        "candidate_policy",
        "metric",
        "training_procedure",
    ):
        if not isinstance(protocol[field], Mapping) or not protocol[field]:
            raise GuardInputError(
                "GUARD_INPUT_VALUE",
                f"Protocol.{field} must be a non-empty object",
                path=f"{path}/{field}",
            )
    _string(
        protocol["metric"].get("name"),
        label="Protocol.metric.name",
        path=f"{path}/metric/name",
    )
    _positive_int(
        protocol["metric"].get("cutoff"),
        label="Protocol.metric.cutoff",
        path=f"{path}/metric/cutoff",
    )
    return protocol


def _validate_claim(value: object) -> dict[str, Any]:
    claim = _object(value, CLAIM_FIELDS, label="Claim", path="/claim")
    for field in (
        "claim_id",
        "protocol_id",
        "claim_kind",
        "target_model",
        "comparator",
        "metric",
    ):
        _string(claim[field], label=f"Claim.{field}", path=f"/claim/{field}")
    _positive_int(
        claim["required_seed_count"],
        label="Claim.required_seed_count",
        path="/claim/required_seed_count",
    )
    if not isinstance(claim["scope"], Mapping) or not claim["scope"]:
        raise GuardInputError(
            "GUARD_INPUT_VALUE",
            "Claim.scope must be a non-empty object",
            path="/claim/scope",
        )
    return claim


def _validate_evidence(value: object) -> dict[str, Any]:
    evidence = _object(
        value, EVIDENCE_FIELDS, label="Current Evidence", path="/current_evidence"
    )
    for field in ("snapshot_id", "claim_id", "protocol_id"):
        _string(
            evidence[field],
            label=f"Current Evidence.{field}",
            path=f"/current_evidence/{field}",
        )
    identifiers = evidence["observation_ids"]
    if not isinstance(identifiers, list) or not all(
        isinstance(item, str) and item for item in identifiers
    ) or len(identifiers) != len(set(identifiers)):
        raise GuardInputError(
            "GUARD_INPUT_VALUE",
            "Current Evidence.observation_ids must be a unique string list",
            path="/current_evidence/observation_ids",
        )
    return evidence


def _validate_proposal(value: object) -> dict[str, Any]:
    proposal = _object(
        value, PROPOSAL_FIELDS, label="Action Proposal", path="/action_proposal"
    )
    for field in (
        "proposal_id",
        "action_family",
        "target_claim_id",
        "protocol_id",
        "target_model",
        "comparator",
        "purpose",
    ):
        _string(
            proposal[field],
            label=f"Action Proposal.{field}",
            path=f"/action_proposal/{field}",
        )
    _positive_int(
        proposal["seed_count"],
        label="Action Proposal.seed_count",
        path="/action_proposal/seed_count",
    )
    seed_ids = proposal["seed_ids"]
    if not isinstance(seed_ids, list) or not all(
        isinstance(item, str) and item for item in seed_ids
    ) or len(seed_ids) != len(set(seed_ids)) or len(seed_ids) != proposal["seed_count"]:
        raise GuardInputError(
            "GUARD_SEED_PLAN_CLOSURE",
            "Action Proposal.seed_ids must contain seed_count unique identities",
            path="/action_proposal/seed_ids",
        )
    proposal["planned_protocol"] = _validate_protocol(
        proposal["planned_protocol"], path="/action_proposal/planned_protocol"
    )
    return proposal


def _validate_observation(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    observation = _object(
        value, OBSERVATION_FIELDS, label="Observation", path="/observation"
    )
    for field in (
        "observation_id",
        "proposal_id",
        "claim_id",
        "protocol_id",
        "target_model",
        "comparator",
        "observation_kind",
        "run_status",
        "artifact_identity_status",
        "evidence_class",
    ):
        _string(
            observation[field],
            label=f"Observation.{field}",
            path=f"/observation/{field}",
        )
    _positive_int(
        observation["seed_count"],
        label="Observation.seed_count",
        path="/observation/seed_count",
    )
    seed_runs = observation["seed_runs"]
    if not isinstance(seed_runs, list) or len(seed_runs) != observation["seed_count"]:
        raise GuardInputError(
            "GUARD_SEED_RUN_CLOSURE",
            "Observation.seed_runs must contain exactly seed_count run identities",
            path="/observation/seed_runs",
        )
    seed_ids: list[str] = []
    run_ids: list[str] = []
    for index, value in enumerate(seed_runs):
        if not isinstance(value, Mapping) or set(value) != {
            "seed_id", "run_id", "artifact_sha256"
        }:
            raise GuardInputError(
                "GUARD_SEED_RUN_CLOSURE",
                "each seed run must bind seed_id, run_id, and artifact_sha256",
                path=f"/observation/seed_runs/{index}",
            )
        seed_ids.append(
            _string(
                value["seed_id"],
                label="Observation.seed_run.seed_id",
                path=f"/observation/seed_runs/{index}/seed_id",
            )
        )
        run_ids.append(
            _string(
                value["run_id"],
                label="Observation.seed_run.run_id",
                path=f"/observation/seed_runs/{index}/run_id",
            )
        )
        _sha256(
            value["artifact_sha256"],
            label="Observation.seed_run.artifact_sha256",
            path=f"/observation/seed_runs/{index}/artifact_sha256",
        )
    if len(seed_ids) != len(set(seed_ids)) or len(run_ids) != len(set(run_ids)):
        raise GuardInputError(
            "GUARD_SEED_RUN_CLOSURE",
            "Observation.seed_runs must use unique seed and run identities",
            path="/observation/seed_runs",
        )
    observation["observed_protocol"] = _validate_protocol(
        observation["observed_protocol"], path="/observation/observed_protocol"
    )
    if not isinstance(observation["metrics"], Mapping):
        raise GuardInputError(
            "GUARD_INPUT_VALUE",
            "Observation.metrics must be an object",
            path="/observation/metrics",
        )
    for key, metric in observation["metrics"].items():
        if not isinstance(key, str) or not key or not isinstance(metric, (int, float)) or isinstance(
            metric, bool
        ) or not math.isfinite(metric):
            raise GuardInputError(
                "GUARD_NONFINITE_OR_INVALID_METRIC",
                "Observation.metrics must map non-empty names to finite numbers",
                path=f"/observation/metrics/{key}",
            )
    if not isinstance(observation["notes"], list) or not all(
        isinstance(item, str) for item in observation["notes"]
    ):
        raise GuardInputError(
            "GUARD_INPUT_VALUE",
            "Observation.notes must be a string list",
            path="/observation/notes",
        )
    return observation


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _diff(left: Any, right: Any, pointer: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        findings: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{pointer}/{_escape(str(key))}"
            if key not in left:
                findings.append({"path": child, "expected": None, "actual": right[key]})
            elif key not in right:
                findings.append({"path": child, "expected": left[key], "actual": None})
            else:
                findings.extend(_diff(left[key], right[key], child))
        return findings
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": pointer, "expected": left, "actual": right}]
        findings: list[dict[str, Any]] = []
        for index, (expected, actual) in enumerate(zip(left, right, strict=True)):
            findings.extend(_diff(expected, actual, f"{pointer}/{index}"))
        return findings
    return [] if left == right else [{"path": pointer, "expected": left, "actual": right}]


def _diagnostic(code: str, message: str, path: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "path": path, **details}


def _base_result(input_digest: str) -> dict[str, Any]:
    return {
        "record_type": "DEVELOPMENT_EVIDENCE_GUARD_RESULT",
        "schema_version": "recclaw.exploration.development_evidence_guard_result.v1",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "input_digest": input_digest,
        "action_legality": {
            "development_verdict": "INCONCLUSIVE",
            "authoritative_action_contract_status": "NOT_STARTED",
            "reason_codes": [],
        },
        "protocol_diagnostics": [],
        "evidence_admissibility": {
            "authoritative_verdict": "INCONCLUSIVE",
            "development_disposition": "NOT_EVALUATED",
            "reason_codes": [],
        },
        "affected_claim_scope": {
            "current_claim_id": None,
            "current_claim_effect": "NONE",
            "protocol_branch_required": False,
            "branch_protocol_digest": None,
        },
        "claim_ceiling": {
            "level": "NO_CLAIM_UPDATE",
            "may_update_search_memory": False,
            "may_update_accepted_evidence_history": False,
            "may_update_claim_state": False,
        },
        "router_directive": "QUARANTINE_INPUT",
        "does_not_create": [
            "ACCEPTED_SPEC",
            "ACTION_CONTRACT",
            "EVIDENCE_ADMISSION_DECISION",
            "ADMISSIBLE_EVENT",
            "CLAIM_TRANSITION",
            "EXECUTION_PERMISSION",
            "PROMOTION",
            "GATE_DECISION",
        ],
    }


def evaluate_evidence_guard(
    *,
    claim: Mapping[str, Any],
    protocol: Mapping[str, Any],
    current_evidence: Mapping[str, Any],
    action_proposal: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one exploration action/observation without authority effects."""

    raw_inputs = {
        "claim": claim,
        "protocol": protocol,
        "current_evidence": current_evidence,
        "action_proposal": action_proposal,
        "observation": observation,
    }
    try:
        input_digest = _digest(raw_inputs)
    except GuardInputError:
        input_digest = "0" * 64
    result = _base_result(input_digest)
    try:
        checked_claim = _validate_claim(claim)
        checked_protocol = _validate_protocol(protocol, path="/protocol")
        checked_evidence = _validate_evidence(current_evidence)
        checked_proposal = _validate_proposal(action_proposal)
        checked_observation = _validate_observation(observation)
    except GuardInputError as exc:
        result["protocol_diagnostics"] = [
            _diagnostic(exc.code, str(exc), exc.path)
        ]
        result["action_legality"]["reason_codes"] = [exc.code]
        result["evidence_admissibility"]["reason_codes"] = [exc.code]
        result["claim_ceiling"]["may_update_search_memory"] = True
        return result

    result["affected_claim_scope"]["current_claim_id"] = checked_claim["claim_id"]
    diagnostics: list[dict[str, Any]] = []
    action_reasons: list[str] = []

    identity_checks = (
        (checked_claim["protocol_id"], checked_protocol["protocol_id"], "GUARD_CLAIM_PROTOCOL_MISMATCH", "/claim/protocol_id"),
        (checked_evidence["claim_id"], checked_claim["claim_id"], "GUARD_EVIDENCE_CLAIM_MISMATCH", "/current_evidence/claim_id"),
        (checked_evidence["protocol_id"], checked_protocol["protocol_id"], "GUARD_EVIDENCE_PROTOCOL_MISMATCH", "/current_evidence/protocol_id"),
        (checked_proposal["target_claim_id"], checked_claim["claim_id"], "GUARD_PROPOSAL_CLAIM_MISMATCH", "/action_proposal/target_claim_id"),
        (checked_proposal["protocol_id"], checked_protocol["protocol_id"], "GUARD_PROPOSAL_PROTOCOL_MISMATCH", "/action_proposal/protocol_id"),
        (checked_proposal["target_model"], checked_claim["target_model"], "GUARD_TARGET_MODEL_MISMATCH", "/action_proposal/target_model"),
        (checked_proposal["comparator"], checked_claim["comparator"], "GUARD_COMPARATOR_MISMATCH", "/action_proposal/comparator"),
    )
    for actual, expected, code, path in identity_checks:
        if actual != expected:
            action_reasons.append(code)
            diagnostics.append(
                _diagnostic(code, "proposal or state identity differs from the current claim scope", path, expected=expected, actual=actual)
            )

    expected_action = PROFILE_ACTION_FAMILY[checked_protocol["profile_family"]]
    if checked_proposal["action_family"] != expected_action:
        action_reasons.append("GUARD_ACTION_FAMILY_UNSUPPORTED")
        diagnostics.append(
            _diagnostic(
                "GUARD_ACTION_FAMILY_UNSUPPORTED",
                "action family is not legal for the current profile family",
                "/action_proposal/action_family",
                expected=expected_action,
                actual=checked_proposal["action_family"],
            )
        )

    claim_metric = checked_claim["metric"]
    protocol_metric = checked_protocol["metric"]["name"]
    if claim_metric != protocol_metric:
        action_reasons.append("GUARD_CLAIM_METRIC_MISMATCH")
        diagnostics.append(
            _diagnostic(
                "GUARD_CLAIM_METRIC_MISMATCH",
                "claim metric differs from the registered protocol metric",
                "/claim/metric",
                expected=protocol_metric,
                actual=claim_metric,
            )
        )

    planned_diff = _diff(checked_protocol, checked_proposal["planned_protocol"])
    for item in planned_diff:
        diagnostics.append(
            _diagnostic(
                "GUARD_PLANNED_PROTOCOL_FLIP",
                "proposal changes the current protocol and requires a separate branch",
                f"/action_proposal/planned_protocol{item['path']}",
                expected=item["expected"],
                actual=item["actual"],
            )
        )
    if planned_diff:
        action_reasons.append("GUARD_PLANNED_PROTOCOL_FLIP")
        result["affected_claim_scope"]["protocol_branch_required"] = True
        result["affected_claim_scope"]["branch_protocol_digest"] = _digest(
            checked_proposal["planned_protocol"]
        )

    if action_reasons:
        result["action_legality"]["development_verdict"] = "ILLEGAL"
        result["action_legality"]["reason_codes"] = sorted(set(action_reasons))
        result["router_directive"] = (
            "ROUTE_TO_PROTOCOL_BRANCH_OR_REVISE"
            if planned_diff
            else "REJECT_PROPOSAL"
        )
    else:
        result["action_legality"]["development_verdict"] = "LEGAL"
        result["router_directive"] = "EXECUTE_DEVELOPMENT_ONLY"

    result["protocol_diagnostics"] = diagnostics
    result["claim_ceiling"]["may_update_search_memory"] = True
    if checked_observation is None:
        return result

    admission_reasons: list[str] = []
    observation_identity_checks = (
        (checked_observation["proposal_id"], checked_proposal["proposal_id"], "GUARD_OBSERVATION_PROPOSAL_MISMATCH"),
        (checked_observation["claim_id"], checked_claim["claim_id"], "GUARD_OBSERVATION_CLAIM_MISMATCH"),
        (checked_observation["protocol_id"], checked_protocol["protocol_id"], "GUARD_OBSERVATION_PROTOCOL_ID_MISMATCH"),
        (checked_observation["target_model"], checked_claim["target_model"], "GUARD_OBSERVATION_MODEL_MISMATCH"),
        (checked_observation["comparator"], checked_claim["comparator"], "GUARD_OBSERVATION_COMPARATOR_MISMATCH"),
        (checked_observation["seed_count"], checked_proposal["seed_count"], "GUARD_SEED_COUNT_MISMATCH"),
        (
            sorted(item["seed_id"] for item in checked_observation["seed_runs"]),
            sorted(checked_proposal["seed_ids"]),
            "GUARD_SEED_IDENTITY_MISMATCH",
        ),
    )
    for actual, expected, code in observation_identity_checks:
        if actual != expected:
            admission_reasons.append(code)

    observed_diff = _diff(checked_protocol, checked_observation["observed_protocol"])
    for item in observed_diff:
        result["protocol_diagnostics"].append(
            _diagnostic(
                "GUARD_OBSERVED_PROTOCOL_FLIP",
                "observation was produced under a different protocol and cannot affect the current claim",
                f"/observation/observed_protocol{item['path']}",
                expected=item["expected"],
                actual=item["actual"],
            )
        )
    if observed_diff:
        admission_reasons.append("GUARD_OBSERVED_PROTOCOL_FLIP")
        result["affected_claim_scope"]["protocol_branch_required"] = True
        result["affected_claim_scope"]["branch_protocol_digest"] = _digest(
            checked_observation["observed_protocol"]
        )

    if checked_observation["observation_id"] in checked_evidence["observation_ids"]:
        admission_reasons.append("GUARD_DUPLICATE_OBSERVATION")
    if checked_observation["evidence_class"] != "DEVELOPMENT_ONLY":
        admission_reasons.append("GUARD_UNSUPPORTED_EVIDENCE_CLASS")
    if result["action_legality"]["development_verdict"] != "LEGAL":
        admission_reasons.append("GUARD_ILLEGAL_ACTION")

    evidence = result["evidence_admissibility"]
    ceiling = result["claim_ceiling"]
    if admission_reasons:
        evidence["development_disposition"] = "EXCLUDE_FROM_CURRENT_CLAIM"
        evidence["reason_codes"] = sorted(set(admission_reasons))
        ceiling["level"] = (
            "CROSS_PROTOCOL_BRANCH_ONLY"
            if observed_diff or planned_diff
            else "NO_CLAIM_UPDATE"
        )
        result["router_directive"] = (
            "ROUTE_TO_PROTOCOL_BRANCH_OR_REVISE"
            if ceiling["level"] == "CROSS_PROTOCOL_BRANCH_ONLY"
            else "QUARANTINE_OBSERVATION"
        )
        return result

    if checked_observation["artifact_identity_status"] != "EXACT":
        evidence["development_disposition"] = "QUARANTINE_PROVENANCE_INCOMPLETE"
        evidence["reason_codes"] = ["GUARD_ARTIFACT_IDENTITY_INCOMPLETE"]
        ceiling["level"] = "PROVENANCE_INCOMPLETE_DIAGNOSTIC"
        result["router_directive"] = "QUARANTINE_OBSERVATION"
        return result

    observation_kind = checked_observation["observation_kind"]
    run_status = checked_observation["run_status"]
    if observation_kind == "INTERFACE_SMOKE" or run_status == "SMOKE_PASS":
        evidence["development_disposition"] = "RECORD_EXECUTABILITY_ONLY"
        evidence["reason_codes"] = ["GUARD_SMOKE_NOT_METRIC_EVIDENCE"]
        ceiling["level"] = "EXECUTABILITY_ONLY"
        result["router_directive"] = "UPDATE_SEARCH_MEMORY_ONLY"
        return result
    if run_status == "RUNTIME_BLOCKED":
        evidence["development_disposition"] = "RECORD_RUNTIME_BLOCKER_ONLY"
        evidence["reason_codes"] = ["GUARD_RUNTIME_BLOCKER_NOT_MECHANISM_RESULT"]
        ceiling["level"] = "RUNTIME_BLOCKER_ONLY"
        result["router_directive"] = "UPDATE_SEARCH_MEMORY_ONLY"
        return result
    if run_status != "SUCCESS" or observation_kind != "METRIC_EVALUATION":
        evidence["development_disposition"] = "RECORD_DIAGNOSTIC_ONLY"
        evidence["reason_codes"] = ["GUARD_NO_SUCCESSFUL_METRIC_OBSERVATION"]
        ceiling["level"] = "NO_CLAIM_UPDATE"
        result["router_directive"] = "UPDATE_SEARCH_MEMORY_ONLY"
        return result
    if claim_metric not in checked_observation["metrics"]:
        evidence["development_disposition"] = "QUARANTINE_METRIC_MISSING"
        evidence["reason_codes"] = ["GUARD_CLAIM_METRIC_MISSING"]
        ceiling["level"] = "NO_CLAIM_UPDATE"
        result["router_directive"] = "QUARANTINE_OBSERVATION"
        return result

    result["affected_claim_scope"]["current_claim_effect"] = "DEVELOPMENT_SIGNAL_ONLY"
    if checked_observation["seed_count"] >= checked_claim["required_seed_count"]:
        evidence["development_disposition"] = (
            "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL"
        )
        ceiling["level"] = "SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL"
    else:
        evidence["development_disposition"] = "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL"
        evidence["reason_codes"] = ["GUARD_SEED_COUNT_BELOW_CLAIM_REQUIREMENT"]
        ceiling["level"] = "LOCAL_SINGLE_OR_FEW_SEED_SIGNAL"
    result["router_directive"] = "UPDATE_SEARCH_MEMORY_ONLY"
    return result


def guard_router_candidates(
    *,
    claim: Mapping[str, Any],
    protocol: Mapping[str, Any],
    current_evidence: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic sidecar decision for a Router candidate pool."""

    decisions: list[dict[str, Any]] = []
    for proposal in proposals:
        result = evaluate_evidence_guard(
            claim=claim,
            protocol=protocol,
            current_evidence=current_evidence,
            action_proposal=proposal,
            observation=None,
        )
        proposal_id = proposal.get("proposal_id") if isinstance(proposal, Mapping) else None
        decisions.append(
            {
                "proposal_id": proposal_id,
                "development_verdict": result["action_legality"]["development_verdict"],
                "router_directive": result["router_directive"],
                "protocol_branch_required": result["affected_claim_scope"][
                    "protocol_branch_required"
                ],
                "reason_codes": result["action_legality"]["reason_codes"],
                "guard_input_digest": result["input_digest"],
            }
        )
    eligible = [
        item["proposal_id"]
        for item in decisions
        if item["development_verdict"] == "LEGAL"
    ]
    return {
        "record_type": "DEVELOPMENT_EVIDENCE_GUARD_ROUTER_BATCH",
        "schema_version": "recclaw.exploration.development_evidence_guard_router_batch.v1",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "candidate_count": len(decisions),
        "eligible_proposal_ids": eligible,
        "decisions": decisions,
        "may_update_search_memory": True,
        "may_update_accepted_evidence_history": False,
        "may_update_claim_state": False,
    }
