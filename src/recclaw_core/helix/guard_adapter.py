"""The only shared/Research adjacency allowed to import Evidence Guard."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from recclaw_evidence_guard.core_v1 import evaluate_evidence_guard
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.mechanism_space.canonical import deep_thaw

from .contracts import (
    CandidateEnvelope,
    GuardContext,
    PortAdjudication,
    PortStage,
    PortStatus,
    RawResultEnvelope,
)
from .ledger import EvidenceGuardLedgerWriterV1


PRE_CALL_ID_BINDING = "FULL_REQUEST_DIGEST_V2"
POST_CALL_ID_BINDING = "FULL_REQUEST_DIGEST_V1"


def _protocol_status(result: dict[str, Any]) -> str:
    if result["affected_claim_scope"]["protocol_branch_required"]:
        return "PROTOCOL_BRANCH"
    return "CURRENT_PROTOCOL"


def _recommended_validation(
    result: dict[str, Any], comparator_delta: float | None
) -> str:
    disposition = result["evidence_admissibility"]["development_disposition"]
    if result["affected_claim_scope"]["protocol_branch_required"]:
        return "SEPARATE_PROTOCOL_VALIDATION"
    if (
        disposition == "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL"
        and comparator_delta is not None
        and comparator_delta > 0.0
    ):
        return "REQUIRES_CONFIRMATION"
    if disposition.startswith("QUARANTINE"):
        return "DIAGNOSTIC_REVIEW"
    return "NONE"


def _adjudication(
    candidate_id: str,
    stage: PortStage,
    result: dict[str, Any],
    *,
    comparator_delta: float | None = None,
) -> PortAdjudication:
    action = result["action_legality"]
    admission = result["evidence_admissibility"]
    reasons = tuple(sorted(set(action["reason_codes"] + admission["reason_codes"])))
    if stage is PortStage.PRE:
        status = PortStatus.ALLOW if action["development_verdict"] == "LEGAL" else PortStatus.BLOCK
        outcome_class = "PRE_RUN_ONLY"
        comparator_delta = None
    else:
        status = PortStatus.ADJUDICATED
        outcome_class = admission["development_disposition"]
    return PortAdjudication(
        candidate_id=candidate_id,
        stage=stage,
        status=status,
        protocol_status=_protocol_status(result),
        outcome_class=outcome_class,
        claim_ceiling=result["claim_ceiling"]["level"],
        reason_codes=reasons,
        comparator_delta=comparator_delta,
        evidence_use=admission["development_disposition"],
        recommended_validation=_recommended_validation(result, comparator_delta),
    )


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def eligible_raw_result(
    payload: Mapping[str, Any],
    *,
    candidate: CandidateEnvelope,
    expected_protocol: Mapping[str, Any],
    metric_name: str,
    require_candidate_id: bool = True,
) -> bool:
    """Shared scientific eligibility predicate for adapter and summary paths."""

    metrics = payload.get("normalized_metrics")
    return bool(
        (not require_candidate_id or payload.get("candidate_id") == candidate.candidate_id)
        and payload.get("opaque_arm_instance_id")
        == candidate.opaque_arm_instance_id
        and payload.get("run_status") == "SUCCESS"
        and payload.get("observation_kind") == "METRIC_EVALUATION"
        and payload.get("artifact_identity_status") == "EXACT"
        and payload.get("target_model") == candidate.target_model
        and payload.get("comparator") == candidate.comparator
        and payload.get("observed_protocol") == expected_protocol
        and isinstance(metrics, Mapping)
        and _finite_metric(metrics.get(metric_name)) is not None
    )


@dataclass(slots=True)
class EvidenceGuardPortV1:
    context: GuardContext
    ledger: EvidenceGuardLedgerWriterV1
    opaque_arm_instance_id: str
    _candidates: dict[str, CandidateEnvelope] = field(default_factory=dict)
    _pre: dict[str, PortAdjudication] = field(default_factory=dict)

    def _evaluation_context(
        self, candidate: CandidateEnvelope
    ) -> dict[str, Any]:
        context = self.context.to_dict()
        claim = dict(context["claim"])
        if claim["target_model"] == "CANDIDATE_SPECIFIC":
            claim["target_model"] = candidate.target_model
            claim["comparator"] = candidate.comparator
        context["claim"] = claim
        snapshot = self.ledger.evidence_snapshot(
            candidate_semantic_digest=candidate.candidate_semantic_digest,
            mechanism_program_digest=candidate.mechanism_program_digest,
            protocol_digest=sha256_digest(
                deep_thaw(candidate.planned_protocol)
            ),
            comparator_identity=candidate.comparator,
        )
        context["current_evidence"] = snapshot.guard_core_projection(
            snapshot_id=snapshot.digest,
            claim_id=str(claim["claim_id"]),
            protocol_id=str(context["protocol"]["protocol_id"]),
        )
        return context

    def _proposal(
        self,
        candidate: CandidateEnvelope,
        *,
        seed_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        claim = deep_thaw(self.context.claim)
        protocol = deep_thaw(self.context.protocol)
        selected_seed_ids = seed_ids or candidate.seed_ids
        return {
            "proposal_id": candidate.candidate_id,
            "action_family": candidate.action_family,
            "target_claim_id": claim["claim_id"],
            "protocol_id": protocol["protocol_id"],
            "planned_protocol": deep_thaw(candidate.planned_protocol),
            "target_model": candidate.target_model,
            "comparator": candidate.comparator,
            "seed_count": len(selected_seed_ids),
            "seed_ids": list(selected_seed_ids),
            "purpose": candidate.purpose,
        }

    def _validation_observation(
        self,
        *,
        candidate: CandidateEnvelope,
        raw_result: RawResultEnvelope,
    ) -> tuple[dict[str, Any], str]:
        protocol_digest = sha256_digest(
            deep_thaw(candidate.planned_protocol)
        )
        prior = self.ledger.raw_results_for_identity(
            candidate_semantic_digest=(
                candidate.candidate_semantic_digest
            ),
            mechanism_program_digest=candidate.mechanism_program_digest,
            protocol_digest=protocol_digest,
            comparator_identity=candidate.comparator,
        )
        current_payload = raw_result.to_dict()
        claim = deep_thaw(self.context.claim)
        metric_name = str(claim["metric"])
        expected_protocol = deep_thaw(candidate.planned_protocol)

        # Failed, cross-protocol, or provenance-incomplete rows remain durable
        # diagnostics, but may never satisfy a scientific seed threshold.
        raw_payloads = (
            tuple(
                payload
                for payload in prior
                if eligible_raw_result(
                    payload,
                    candidate=candidate,
                    expected_protocol=expected_protocol,
                    metric_name=metric_name,
                    require_candidate_id=False,
                )
            )
            + (current_payload,)
            if eligible_raw_result(
                current_payload,
                candidate=candidate,
                expected_protocol=expected_protocol,
                metric_name=metric_name,
            )
            else (current_payload,)
        )
        seed_runs: dict[str, dict[str, Any]] = {}
        seed_metrics: dict[str, dict[str, Any]] = {}
        metric_values: dict[str, list[tuple[float, int]]] = {}
        statuses: list[str] = []
        artifact_statuses: list[str] = []
        for payload in raw_payloads:
            statuses.append(str(payload["run_status"]))
            artifact_statuses.append(str(payload["artifact_identity_status"]))
            payload_metrics = dict(payload["normalized_metrics"])
            new_seed_count = 0
            for seed_run in payload["seed_runs"]:
                seed_id = str(seed_run["seed_id"])
                prior_seed = seed_runs.get(seed_id)
                if prior_seed is not None:
                    # Fixed-slot campaigns may execute the same semantic
                    # candidate on the same frozen seed more than once. A
                    # fresh run artifact is not a new scientific seed and
                    # must not increase evidence weight. Accept only an exact
                    # normalized-metric repeat; a changed result for the same
                    # seed remains fail-closed.
                    if seed_metrics[seed_id] != payload_metrics:
                        raise ValueError(
                            "validation seed identity substitution"
                        )
                    continue
                seed_runs[seed_id] = dict(seed_run)
                seed_metrics[seed_id] = payload_metrics
                new_seed_count += 1
            if new_seed_count:
                for key, value in payload_metrics.items():
                    metric_values.setdefault(str(key), []).append(
                        (float(value), new_seed_count)
                    )
        ordered_seed_runs = [
            seed_runs[key] for key in sorted(seed_runs)
        ]
        metrics = {
            key: (
                sum(value * weight for value, weight in values)
                / sum(weight for _value, weight in values)
            )
            for key, values in sorted(metric_values.items())
        }
        observation_id = sha256_digest(
            {
                "candidate_semantic_digest": (
                    candidate.candidate_semantic_digest
                ),
                "mechanism_program_digest": candidate.mechanism_program_digest,
                "comparator": candidate.comparator,
                "protocol_digest": protocol_digest,
                "seed_runs": ordered_seed_runs,
            }
        )
        observation = {
            "observation_id": observation_id,
            "proposal_id": candidate.candidate_id,
            "claim_id": deep_thaw(self.context.claim)["claim_id"],
            "protocol_id": deep_thaw(self.context.protocol)[
                "protocol_id"
            ],
            "observed_protocol": deep_thaw(raw_result.observed_protocol),
            "target_model": raw_result.target_model,
            "comparator": raw_result.comparator,
            "seed_count": len(ordered_seed_runs),
            "seed_runs": ordered_seed_runs,
            "observation_kind": raw_result.observation_kind,
            "run_status": (
                "SUCCESS"
                if all(status == "SUCCESS" for status in statuses)
                else str(raw_result.run_status)
            ),
            "artifact_identity_status": (
                "EXACT"
                if all(status == "EXACT" for status in artifact_statuses)
                else str(raw_result.artifact_identity_status)
            ),
            "evidence_class": "DEVELOPMENT_ONLY",
            "metrics": metrics,
            "notes": [],
        }
        return observation, protocol_digest

    def pre_run(self, candidate: CandidateEnvelope) -> PortAdjudication:
        if candidate.opaque_arm_instance_id != self.opaque_arm_instance_id:
            raise ValueError("wrong-arm CandidateEnvelope substitution")
        context = self._evaluation_context(candidate)
        request = {
            "candidate": candidate.to_dict(),
            "context": context,
            "phase": "PRE",
        }
        result = evaluate_evidence_guard(
            claim=context["claim"],
            protocol=context["protocol"],
            current_evidence=context["current_evidence"],
            action_proposal=self._proposal(candidate),
            observation=None,
        )
        call_id = (
            f"pre:{candidate.candidate_id}:" + sha256_digest(request)
        )
        stored, _created = self.ledger.commit_create_once(
            guard_call_id=call_id,
            phase="PRE",
            candidate_id=candidate.candidate_id,
            request=request,
            full_event=result,
        )
        adjudication = _adjudication(candidate.candidate_id, PortStage.PRE, stored)
        self._candidates[candidate.candidate_id] = candidate
        self._pre[candidate.candidate_id] = adjudication
        return adjudication

    def post_run(self, raw_result: RawResultEnvelope) -> PortAdjudication:
        candidate = self._candidates.get(raw_result.candidate_id)
        pre = self._pre.get(raw_result.candidate_id)
        if candidate is None or pre is None:
            recovered = self.ledger.request_and_event_for_candidate(
                phase="PRE", candidate_id=raw_result.candidate_id
            )
            if recovered is not None:
                request, event = recovered
                candidate_payload = request["candidate"]
                candidate = CandidateEnvelope(
                    candidate_id=candidate_payload["candidate_id"],
                    candidate_semantic_digest=candidate_payload[
                        "candidate_semantic_digest"
                    ],
                    opaque_arm_instance_id=candidate_payload[
                        "opaque_arm_instance_id"
                    ],
                    common_status=candidate_payload["common_status"],
                    mechanism_program_digest=candidate_payload[
                        "mechanism_program_digest"
                    ],
                    common_plan_digest=candidate_payload["common_plan_digest"],
                    action_family=candidate_payload["action_family"],
                    planned_protocol=candidate_payload["planned_protocol"],
                    target_model=candidate_payload["target_model"],
                    comparator=candidate_payload["comparator"],
                    seed_ids=tuple(candidate_payload["seed_ids"]),
                    purpose=candidate_payload["purpose"],
                )
                current_context = self._evaluation_context(candidate)
                if (
                    request["context"]["claim"]
                    != current_context["claim"]
                    or request["context"]["protocol"]
                    != current_context["protocol"]
                ):
                    raise ValueError(
                        "recovered PRE authority context mismatch"
                    )
                pre = _adjudication(candidate.candidate_id, PortStage.PRE, event)
                self._candidates[candidate.candidate_id] = candidate
                self._pre[candidate.candidate_id] = pre
        if candidate is None or pre is None or pre.status is not PortStatus.ALLOW:
            raise ValueError("POST requires a matching allowed PRE call")
        if (
            candidate.opaque_arm_instance_id != self.opaque_arm_instance_id
            or raw_result.opaque_arm_instance_id != self.opaque_arm_instance_id
        ):
            raise ValueError("wrong-arm RawResultEnvelope substitution")
        context = self._evaluation_context(candidate)
        observation, protocol_digest = self._validation_observation(
            candidate=candidate,
            raw_result=raw_result,
        )
        request = {
            "candidate_digest": candidate.digest,
            "candidate_semantic_digest": candidate.candidate_semantic_digest,
            "mechanism_program_digest": candidate.mechanism_program_digest,
            "protocol_digest": protocol_digest,
            "comparator_identity": candidate.comparator,
            "context": context,
            "phase": "POST",
            "raw_result": raw_result.to_dict(),
        }
        result = evaluate_evidence_guard(
            claim=context["claim"],
            protocol=context["protocol"],
            current_evidence=context["current_evidence"],
            action_proposal=self._proposal(
                candidate,
                seed_ids=tuple(
                    str(item["seed_id"])
                    for item in observation["seed_runs"]
                ),
            ),
            observation=observation,
        )
        comparator_delta = None
        scope = deep_thaw(self.context.claim).get("scope")
        comparator_metric = (
            _finite_metric(scope.get("frozen_comparator_ndcg_at_10"))
            if isinstance(scope, Mapping)
            else None
        )
        candidate_metric = _finite_metric(
            observation["metrics"].get(
                str(deep_thaw(self.context.claim)["metric"])
            )
        )
        if (
            result["evidence_admissibility"]["development_disposition"].startswith(
                "COUNT_AS_"
            )
            and comparator_metric is not None
            and candidate_metric is not None
        ):
            comparator_delta = candidate_metric - comparator_metric
        call_id = f"post:{candidate.candidate_id}:{sha256_digest(request)}"
        stored, _created = self.ledger.commit_create_once(
            guard_call_id=call_id,
            phase="POST",
            candidate_id=candidate.candidate_id,
            request=request,
            full_event=result,
        )
        for seed_run in raw_result.seed_runs:
            self.ledger.record_evidence_observation(
                candidate_semantic_digest=(
                    candidate.candidate_semantic_digest
                ),
                mechanism_program_digest=candidate.mechanism_program_digest,
                protocol_digest=protocol_digest,
                comparator_identity=candidate.comparator,
                observation_seed=str(seed_run["seed_id"]),
                observation_id=str(observation["observation_id"]),
                raw_result=raw_result.to_dict(),
            )
        return _adjudication(
            candidate.candidate_id,
            PortStage.POST,
            stored,
            comparator_delta=comparator_delta,
        )

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "adapter": "EvidenceGuardPortV1",
                "context_digest": self.context.digest,
                "core_sha256": "d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af",
                "ledger_namespace": self.ledger.namespace,
                "pre_call_id_binding": PRE_CALL_ID_BINDING,
                "post_call_id_binding": POST_CALL_ID_BINDING,
            }
        )
