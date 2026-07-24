"""Three-phase CommonExecutionGuardV1 for the M1 fake executable slice."""

from __future__ import annotations

import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.mechanism_space import CompileReportV1, CompileStatus, compile_program

from .canonical import bytes_sha256, canonical_json_bytes, sha256_digest
from .contracts import ResourceCeilingsV1
from .materialization import (
    classify_execution_trust,
    verify_binding_v2,
    verify_materialization,
)
from .runtime_contracts import (
    CandidateExecutionBindingV2,
    CommonDecision,
    CommonEligibleActionV1,
    CommonExecutionPermitV1,
    CommonPlanDecisionV1,
    CommonPreExecutionDecisionV1,
    CommonResultClosureV1,
    DevelopmentExecutionGateDecisionV1,
    DevelopmentRecSysProtocolV1,
    ExecutionStartReceiptV1,
    ExecutionTrustClassificationV1,
    GateStatus,
    MaterializationReportV1,
    RawResultEnvelopeV1,
    RawRunOutputV1,
    StartStatus,
    TrustClass,
    create_common_eligible_action,
)
from .runtime_release import (
    common_guard_policy,
    common_release_projection_digest,
    development_protocol,
    executable_profile_digest,
    profile_supports,
    runtime_release_contract,
    source_manifest,
    source_manifest_digest,
    validate_frozen_environment,
)


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]{2,127}$")


def _subcheck(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {
        "detail": detail,
        "name": name,
        "status": "PASS" if passed else "FAIL",
    }


def _compile_dict(report: CompileReportV1 | Mapping[str, Any]) -> dict[str, Any]:
    return report.to_dict() if isinstance(report, CompileReportV1) else dict(report)


def _decision(
    failures: list[str], *, inconclusive: bool = False
) -> CommonDecision:
    if inconclusive:
        return CommonDecision.INCONCLUSIVE
    return CommonDecision.DENY if failures else CommonDecision.PASS


class CommonExecutionGuardV1:
    """Mechanical common guard. It contains no Arm or Evidence Guard branch."""

    phase_schedule = ("plan_check", "pre_execute", "close_result")

    def __init__(self) -> None:
        policy = common_guard_policy()
        if tuple(policy["phase_schedule"]) != self.phase_schedule:
            raise ValueError("package policy has a different phase schedule")
        self.release_projection_digest = common_release_projection_digest()

    def plan_check(
        self,
        *,
        program: Mapping[str, Any],
        caller_compile_report: CompileReportV1 | Mapping[str, Any],
        protocol: DevelopmentRecSysProtocolV1,
        budget: ResourceCeilingsV1,
    ) -> tuple[CommonPlanDecisionV1, CommonEligibleActionV1 | None]:
        policy = common_guard_policy()
        subchecks: list[dict[str, Any]] = []
        failures: list[str] = []
        fresh = compile_program(program)
        caller = _compile_dict(caller_compile_report)

        schema_ok = fresh.status is not CompileStatus.INVALID
        subchecks.append(_subcheck("SCHEMA", schema_ok))
        if not schema_ok:
            failures.append("SCHEMA_INVALID")

        compile_ok = (
            fresh.status in {CompileStatus.VALID_WIRED, CompileStatus.VALID_NEEDS_IMPLEMENTATION}
            and fresh.to_dict() == caller
        )
        subchecks.append(
            _subcheck(
                "BL_COMPILE",
                compile_ok,
                "" if compile_ok else "fresh compiler output differs or is not valid",
            )
        )
        if not compile_ok:
            failures.append("BL_COMPILE_FAILED")

        protocol_ok = protocol.to_dict() == development_protocol().to_dict()
        subchecks.append(_subcheck("PROTOCOL", protocol_ok))
        if not protocol_ok:
            failures.append("FROZEN_PROTOCOL_CONTRACT_MISMATCH")

        capability_ok, capability_detail = profile_supports(program)
        subchecks.append(
            _subcheck("CAPABILITY", capability_ok, ",".join(capability_detail))
        )
        if not capability_ok:
            failures.append("CAPABILITY_UNSUPPORTED")

        candidate_id = str(fresh.candidate_id or caller.get("candidate_id") or "")
        path_ok = bool(_SAFE_ID.fullmatch(candidate_id))
        subchecks.append(_subcheck("PATH_PLAN", path_ok))
        if not path_ok:
            failures.append("PATH_OUT_OF_SCOPE")

        budget_ok = (
            budget.ordinary_executions == 1
            and budget.common_validation_count >= 1
        )
        subchecks.append(_subcheck("BUDGET", budget_ok))
        if not budget_ok:
            failures.append("BUDGET_DENIED")

        environment_failures = validate_frozen_environment()
        runner_ok = (
            not environment_failures
            and runtime_release_contract()["runner_abi"]
            == "recclaw.fake-non-training-runner.v1"
        )
        subchecks.append(
            _subcheck("RUNNER_ABI", runner_ok, ",".join(environment_failures))
        )
        if not runner_ok:
            failures.append("RUNNER_IDENTITY_MISMATCH")

        reason_codes = tuple(sorted(set(failures)))
        unknown = set(reason_codes) - set(policy["reason_codes"])
        if unknown:
            raise ValueError(f"unregistered common reason codes: {sorted(unknown)}")
        compile_report_digest = sha256_digest(caller)
        plan = CommonPlanDecisionV1(
            {
                "candidate_id": candidate_id,
                "compile_report_digest": compile_report_digest,
                "decision": _decision(failures).value,
                "mechanism_program_digest": fresh.mechanism_program_digest,
                "mechanism_semantics_digest": fresh.mechanism_semantics_digest,
                "profile_digest": executable_profile_digest(),
                "protocol_digest": protocol.digest,
                "reason_codes": reason_codes,
                "release_projection_digest": self.release_projection_digest,
                "subchecks": subchecks,
            }
        )
        if failures:
            return plan, None
        eligible = create_common_eligible_action(
            {
                "budget_digest": sha256_digest(budget.to_dict()),
                "candidate_id": candidate_id,
                "compile_projection": {
                    "candidate_id": fresh.candidate_id,
                    "mechanism_program_digest": fresh.mechanism_program_digest,
                    "mechanism_semantics_digest": fresh.mechanism_semantics_digest,
                    "required_capabilities": list(fresh.required_capabilities),
                    "space_identity": (
                        fresh.space_identity.to_dict() if fresh.space_identity else None
                    ),
                    "status": fresh.status.value,
                },
                "compile_report_digest": compile_report_digest,
                "plan_decision_digest": plan.digest,
                "program_digest": fresh.mechanism_program_digest,
                "protocol_digest": protocol.digest,
                "release_projection_digest": self.release_projection_digest,
            }
        )
        return plan, eligible

    def pre_execute(
        self,
        *,
        eligible: CommonEligibleActionV1,
        report: MaterializationReportV1,
        trust: ExecutionTrustClassificationV1,
        binding: CandidateExecutionBindingV2,
        gate: DevelopmentExecutionGateDecisionV1,
    ) -> tuple[CommonPreExecutionDecisionV1, CommonExecutionPermitV1 | None]:
        subchecks: list[dict[str, Any]] = []
        failures: list[str] = []
        root = Path(str(binding.arm_private_root))

        materialization_identity_ok = (
            report.status == "MATERIALIZED"
            and report.candidate_id == eligible.candidate_id
            and report.compile_report_digest == eligible.compile_report_digest
        )
        materialization_bytes_ok, materialization_reasons = verify_materialization(
            report, eligible=eligible, arm_runtime_root=root
        )
        materialization_ok = materialization_identity_ok and materialization_bytes_ok
        subchecks.append(
            _subcheck(
                "MATERIALIZATION",
                materialization_ok,
                ",".join(materialization_reasons),
            )
        )
        if not materialization_ok:
            failures.append("MATERIALIZATION_INVALID")

        observed_trust = classify_execution_trust(report, arm_runtime_root=root)
        path_ok = observed_trust.to_dict() == trust.to_dict()
        subchecks.append(_subcheck("PATH_CLOSURE", path_ok))
        if not path_ok:
            failures.append("PATH_OUT_OF_SCOPE")

        import_ok = False
        import_detail = ""
        try:
            module_name, attribute = str(report.entrypoint).split(":", 1)
            imported = importlib.import_module(module_name)
            handler = getattr(imported, attribute)
            imported_digest = bytes_sha256(Path(imported.__file__).read_bytes())
            import_ok = callable(handler) and imported_digest in {
                item["sha256"] for item in source_manifest()
            }
            import_detail = imported_digest
        except (ImportError, AttributeError, OSError, ValueError):
            handler = None
        subchecks.append(_subcheck("IMPORT_ATTESTATION", import_ok, import_detail))
        if not import_ok:
            failures.append("IMPORT_FAILED")

        smoke_ok = False
        smoke_detail = ""
        if import_ok and handler is not None:
            config_row = next(
                (
                    row
                    for row in report.files
                    if str(row["path"]).endswith("/handler_config.json")
                ),
                None,
            )
            if config_row is not None:
                try:
                    config_path = root.joinpath(*str(config_row["path"]).split("/"))
                    config_bytes = config_path.read_bytes()
                    config = json.loads(config_bytes)
                    smoke = handler(config)
                    smoke_ok = (
                        smoke["optimizer_steps"] == 0
                        and smoke["training_backend_started"] is False
                        and math.isfinite(float(smoke["interface_loss"]))
                    )
                    if smoke_ok:
                        smoke_detail = sha256_digest(
                            {
                                "config_sha256": bytes_sha256(config_bytes),
                                "handler_source_sha256": import_detail,
                                "smoke_output": smoke,
                            }
                        )
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    smoke_ok = False
        subchecks.append(
            _subcheck("NON_TRAINING_SMOKE_ATTESTATION", smoke_ok, smoke_detail)
        )
        if not smoke_ok:
            failures.append("SMOKE_FAILED")

        trust_ok = trust.classification == TrustClass.PACKAGE_OWNED_TYPED_TEMPLATE.value
        subchecks.append(_subcheck("TRUST_CLASSIFICATION", trust_ok))
        if not trust_ok:
            failures.append("TRUST_CLASSIFICATION_DENIED")

        binding_ok, binding_reasons = verify_binding_v2(
            binding, eligible=eligible, report=report, trust=trust
        )
        subchecks.append(_subcheck("BINDING_V2", binding_ok, ",".join(binding_reasons)))
        if not binding_ok:
            failures.append("BINDING_IDENTITY_MISMATCH")

        budget_ok = binding.budget_digest == eligible.budget_digest
        subchecks.append(_subcheck("BUDGET", budget_ok))
        if not budget_ok:
            failures.append("BUDGET_DENIED")

        gate_ok = (
            gate.decision == GateStatus.ALLOW.value
            and gate.binding_digest == binding.digest
            and gate.materialization_digest == report.digest
            and gate.trust_classification_digest == trust.digest
        )
        subchecks.append(_subcheck("DEVELOPMENT_EXECUTION_GATE", gate_ok))
        if not gate_ok:
            failures.append("TRUST_CLASSIFICATION_DENIED")

        decision = CommonPreExecutionDecisionV1(
            {
                "binding_digest": binding.digest,
                "decision": _decision(failures).value,
                "gate_decision_digest": gate.digest,
                "materialization_digest": report.digest,
                "plan_decision_digest": eligible.plan_decision_digest,
                "reason_codes": sorted(set(failures)),
                "release_projection_digest": self.release_projection_digest,
                "subchecks": subchecks,
            }
        )
        if failures:
            return decision, None
        permit = CommonExecutionPermitV1(
            {
                "backend_digest": source_manifest_digest(),
                "binding_digest": binding.digest,
                "budget_digest": binding.budget_digest,
                "candidate_id": binding.candidate_id,
                "gate_decision_digest": gate.digest,
                "ordinary_launch_attempt_ordinal": 1,
                "pre_execution_decision_digest": decision.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
            }
        )
        return decision, permit

    def close_result(
        self,
        *,
        permit: CommonExecutionPermitV1,
        binding: CandidateExecutionBindingV2,
        claim: Mapping[str, Any],
        receipt: ExecutionStartReceiptV1,
        raw_output: RawRunOutputV1,
        artifact_closure: list[Mapping[str, Any]],
    ) -> tuple[CommonResultClosureV1, RawResultEnvelopeV1 | None]:
        subchecks: list[dict[str, Any]] = []
        failures: list[str] = []

        permit_ok = (
            permit.binding_digest == binding.digest
            and permit.round_id == binding.round_id
            and permit.run_id == binding.run_id
        )
        subchecks.append(_subcheck("PERMIT", permit_ok))
        if not permit_ok:
            failures.append("RESULT_CLOSURE_FAILED")

        claim_ok = (
            claim.get("permit_digest") == permit.digest
            and claim.get("binding_digest") == binding.digest
            and claim.get("round_id") == binding.round_id
            and claim.get("claim_state") == "FINISHED"
            and claim.get("execution_debited") == 1
        )
        subchecks.append(_subcheck("CLAIM", claim_ok))
        if not claim_ok:
            failures.append("RESULT_CLOSURE_FAILED")

        receipt_ok = (
            receipt.start_status == StartStatus.STARTED.value
            and receipt.claim_id == claim.get("claim_id")
            and receipt.permit_digest == permit.digest
            and receipt.binding_digest == binding.digest
        )
        subchecks.append(_subcheck("START_RECEIPT", receipt_ok))
        if not receipt_ok:
            failures.append("RESULT_CLOSURE_FAILED")

        runner_ok = (
            raw_output.runner_abi == runtime_release_contract()["runner_abi"]
            and raw_output.permit_digest == permit.digest
            and raw_output.binding_digest == binding.digest
        )
        subchecks.append(_subcheck("RUNNER_IDENTITY", runner_ok))
        if not runner_ok:
            failures.append("RUNNER_IDENTITY_MISMATCH")

        receipt_rows = [
            row
            for row in artifact_closure
            if row.get("artifact_type") == "EXECUTION_START_RECEIPT_V1"
        ]
        raw_rows = [
            row
            for row in artifact_closure
            if row.get("artifact_type") == "RAW_RUN_OUTPUT_V1"
        ]
        artifact_ok = (
            len(receipt_rows) == 1
            and len(raw_rows) == 1
            and receipt_rows[0].get("round_id") == binding.round_id
            and raw_rows[0].get("round_id") == binding.round_id
            and receipt_rows[0].get("sha256")
            == bytes_sha256(canonical_json_bytes(receipt.to_dict()))
            and raw_rows[0].get("sha256")
            == bytes_sha256(canonical_json_bytes(raw_output.to_dict()))
        )
        raw_ok = (
            raw_output.exit_status == "SUCCESS"
            and raw_output.optimizer_steps == 0
            and raw_output.training_backend_started is False
            and raw_output.evaluation_purpose == "NON_OUTCOME_BEARING_INTERFACE_SMOKE"
            and raw_output.normalized_metrics == {}
            and artifact_ok
        )
        subchecks.append(_subcheck("RAW_OUTPUT", raw_ok))
        if not raw_ok:
            failures.append("RESULT_CLOSURE_FAILED")

        finite_ok = math.isfinite(float(raw_output.interface_loss))
        subchecks.append(_subcheck("FINITE_SERIALIZATION", finite_ok))
        if not finite_ok:
            failures.append("RESULT_CLOSURE_FAILED")

        budget_ok = permit.budget_digest == binding.budget_digest
        subchecks.append(_subcheck("BUDGET_CLOSURE", budget_ok))
        if not budget_ok:
            failures.append("BUDGET_DENIED")

        closure = CommonResultClosureV1(
            {
                "claim_id": str(claim.get("claim_id") or ""),
                "decision": _decision(failures).value,
                "permit_digest": permit.digest,
                "raw_output_digest": raw_output.digest,
                "reason_codes": sorted(set(failures)),
                "release_projection_digest": self.release_projection_digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "start_receipt_digest": receipt.digest,
                "subchecks": subchecks,
            }
        )
        if failures:
            return closure, None
        envelope = RawResultEnvelopeV1(
            {
                "artifact_closure": artifact_closure,
                "binding_digest": binding.digest,
                "candidate_id": binding.candidate_id,
                "common_result_closure_digest": closure.digest,
                "evaluation_purpose": "NON_OUTCOME_BEARING_INTERFACE_SMOKE",
                "exit_status": raw_output.exit_status,
                "metric_source": "NONE_NON_OUTCOME_BEARING_SMOKE",
                "normalized_metrics": {},
                "ordinary_execution_start_index": 1,
                "partition_role": "NOT_APPLICABLE_NO_DATA_READ",
                "raw_output_digest": raw_output.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "seed": binding.search_seed,
            }
        )
        return closure, envelope


__all__ = ["CommonExecutionGuardV1"]
