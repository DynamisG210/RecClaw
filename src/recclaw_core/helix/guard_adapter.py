"""The only shared/Research adjacency allowed to import Evidence Guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


def _protocol_status(result: dict[str, Any]) -> str:
    if result["affected_claim_scope"]["protocol_branch_required"]:
        return "PROTOCOL_BRANCH"
    return "CURRENT_PROTOCOL"


def _recommended_validation(result: dict[str, Any]) -> str:
    disposition = result["evidence_admissibility"]["development_disposition"]
    if result["affected_claim_scope"]["protocol_branch_required"]:
        return "SEPARATE_PROTOCOL_VALIDATION"
    if disposition in {
        "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
        "RECORD_EXECUTABILITY_ONLY",
    }:
        return "REQUIRES_CONFIRMATION"
    if disposition.startswith("QUARANTINE"):
        return "DIAGNOSTIC_REVIEW"
    return "NONE"


def _adjudication(
    candidate_id: str,
    stage: PortStage,
    result: dict[str, Any],
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
        comparator_delta = None
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
        recommended_validation=_recommended_validation(result),
    )


@dataclass(slots=True)
class EvidenceGuardPortV1:
    context: GuardContext
    ledger: EvidenceGuardLedgerWriterV1
    opaque_arm_instance_id: str
    _candidates: dict[str, CandidateEnvelope] = field(default_factory=dict)
    _pre: dict[str, PortAdjudication] = field(default_factory=dict)

    def _proposal(self, candidate: CandidateEnvelope) -> dict[str, Any]:
        claim = deep_thaw(self.context.claim)
        protocol = deep_thaw(self.context.protocol)
        return {
            "proposal_id": candidate.candidate_id,
            "action_family": candidate.action_family,
            "target_claim_id": claim["claim_id"],
            "protocol_id": protocol["protocol_id"],
            "planned_protocol": deep_thaw(candidate.planned_protocol),
            "target_model": candidate.target_model,
            "comparator": candidate.comparator,
            "seed_count": len(candidate.seed_ids),
            "seed_ids": list(candidate.seed_ids),
            "purpose": candidate.purpose,
        }

    def pre_run(self, candidate: CandidateEnvelope) -> PortAdjudication:
        if candidate.opaque_arm_instance_id != self.opaque_arm_instance_id:
            raise ValueError("wrong-arm CandidateEnvelope substitution")
        request = {
            "candidate": candidate.to_dict(),
            "context": self.context.to_dict(),
            "phase": "PRE",
        }
        result = evaluate_evidence_guard(
            claim=deep_thaw(self.context.claim),
            protocol=deep_thaw(self.context.protocol),
            current_evidence=deep_thaw(self.context.current_evidence),
            action_proposal=self._proposal(candidate),
            observation=None,
        )
        call_id = f"pre:{candidate.candidate_id}"
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
                if request["context"] != self.context.to_dict():
                    raise ValueError("recovered PRE context does not match this adapter")
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
        observation = {
            "observation_id": raw_result.raw_result_digest,
            "proposal_id": candidate.candidate_id,
            "claim_id": deep_thaw(self.context.claim)["claim_id"],
            "protocol_id": deep_thaw(self.context.protocol)["protocol_id"],
            "observed_protocol": deep_thaw(raw_result.observed_protocol),
            "target_model": raw_result.target_model,
            "comparator": raw_result.comparator,
            "seed_count": len(raw_result.seed_runs),
            "seed_runs": [deep_thaw(item) for item in raw_result.seed_runs],
            "observation_kind": raw_result.observation_kind,
            "run_status": raw_result.run_status,
            "artifact_identity_status": raw_result.artifact_identity_status,
            "evidence_class": "DEVELOPMENT_ONLY",
            "metrics": deep_thaw(raw_result.normalized_metrics),
            "notes": [],
        }
        request = {
            "candidate_digest": candidate.digest,
            "context": self.context.to_dict(),
            "phase": "POST",
            "raw_result": raw_result.to_dict(),
        }
        result = evaluate_evidence_guard(
            claim=deep_thaw(self.context.claim),
            protocol=deep_thaw(self.context.protocol),
            current_evidence=deep_thaw(self.context.current_evidence),
            action_proposal=self._proposal(candidate),
            observation=observation,
        )
        call_id = f"post:{candidate.candidate_id}:{raw_result.raw_result_digest}"
        stored, _created = self.ledger.commit_create_once(
            guard_call_id=call_id,
            phase="POST",
            candidate_id=candidate.candidate_id,
            request=request,
            full_event=result,
        )
        return _adjudication(candidate.candidate_id, PortStage.POST, stored)

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "adapter": "EvidenceGuardPortV1",
                "context_digest": self.context.digest,
                "core_sha256": "d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af",
                "ledger_namespace": self.ledger.namespace,
            }
        )
