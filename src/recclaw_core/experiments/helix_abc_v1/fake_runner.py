"""Package-owned launcher and fake, non-training M1 runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes
from .runtime_contracts import (
    CandidateExecutionBindingV2,
    CommonDecision,
    CommonExecutionPermitV1,
    CommonPreExecutionDecisionV1,
    DevelopmentExecutionGateDecisionV1,
    ExecutionStartReceiptV1,
    GateStatus,
    RawResultEnvelopeV1,
    RawRunOutputV1,
    StartStatus,
)
from .runtime_handlers import run_non_training_smoke
from .runtime_release import runtime_release_contract
from .state_store import (
    MarkExecutionFinishedCommand,
    MarkExecutionStartedCommand,
    RegisterArtifactCommand,
    SingleWriterExperimentStoreV1,
)


class FakeNonTrainingRunnerV1:
    """Pure runner: no state-store writer, dataset, optimizer, or metric access."""

    runner_abi = "recclaw.fake-non-training-runner.v1"

    def run(
        self,
        *,
        permit: CommonExecutionPermitV1,
        binding: CandidateExecutionBindingV2,
        gate: DevelopmentExecutionGateDecisionV1,
        pre_execution: CommonPreExecutionDecisionV1,
    ) -> RawRunOutputV1:
        if gate.decision != GateStatus.ALLOW.value:
            raise ValueError("Runner requires an ALLOW development gate decision")
        if pre_execution.decision != CommonDecision.PASS.value:
            raise ValueError("Runner requires a COMMON_PASS pre-execution decision")
        if (
            permit.gate_decision_digest != gate.digest
            or permit.pre_execution_decision_digest != pre_execution.digest
            or permit.binding_digest != binding.digest
        ):
            raise ValueError("Runner decision/permit scope mismatch")
        if permit.runner_abi != self.runner_abi or binding.runner_abi != self.runner_abi:
            raise ValueError("Runner ABI mismatch")

        root = Path(str(binding.arm_private_root))
        config_path = (
            root
            / "recclaw_ext"
            / "generated"
            / str(binding.candidate_id)
            / "handler_config.json"
        )
        config = json.loads(config_path.read_bytes())
        smoke = run_non_training_smoke(config)
        return RawRunOutputV1(
            {
                "binding_digest": binding.digest,
                "candidate_id": binding.candidate_id,
                "checks": smoke["checks"],
                "evaluation_purpose": "NON_OUTCOME_BEARING_INTERFACE_SMOKE",
                "exit_status": "SUCCESS",
                "interface_loss": smoke["interface_loss"],
                "mechanism_axes_exercised": smoke["mechanism_axes_exercised"],
                "normalized_metrics": {},
                "optimizer_steps": smoke["optimizer_steps"],
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": self.runner_abi,
                "training_backend_started": smoke["training_backend_started"],
            }
        )


class PackageOwnedLauncherV1:
    """Orders committed claim, start receipt, runner call, and raw output."""

    def __init__(self, store: SingleWriterExperimentStoreV1) -> None:
        self._store = store
        self.runner_launch_count = 0

    def launch(
        self,
        *,
        permit: CommonExecutionPermitV1,
        binding: CandidateExecutionBindingV2,
        gate: DevelopmentExecutionGateDecisionV1,
        pre_execution: CommonPreExecutionDecisionV1,
    ) -> tuple[
        ExecutionStartReceiptV1,
        RawRunOutputV1,
        tuple[dict[str, Any], dict[str, Any]],
    ]:
        claim = self._store.get_execution_claim(str(binding.round_id))
        if (
            claim["claim_state"] != "CLAIMED"
            or claim["permit_digest"] != permit.digest
            or claim["binding_digest"] != binding.digest
        ):
            raise ValueError("Launcher requires the exact committed, unused execution claim")
        receipt = ExecutionStartReceiptV1(
            {
                "binding_digest": binding.digest,
                "claim_id": claim["claim_id"],
                "ordinary_launch_attempt_ordinal": 1,
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": runtime_release_contract()["runner_abi"],
                "start_status": StartStatus.STARTED.value,
            }
        )
        receipt_path = f"artifacts/{binding.run_id}/execution_start_receipt.v1.json"
        receipt_artifact = self._store.register_artifact(
            RegisterArtifactCommand(
                round_id=str(binding.round_id),
                artifact_type="EXECUTION_START_RECEIPT_V1",
                relative_path=receipt_path,
                producer="PackageOwnedLauncherV1",
                idempotency_key=f"receipt:{claim['claim_id']}",
            ),
            canonical_json_bytes(receipt.to_dict()),
        )
        self._store.mark_execution_started(
            MarkExecutionStartedCommand(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                receipt_artifact_id=str(receipt_artifact["artifact_id"]),
                idempotency_key=(
                    f"execution-start:{claim['claim_id']}:{receipt_artifact['artifact_id']}"
                ),
            )
        )
        self.runner_launch_count += 1
        raw_output = FakeNonTrainingRunnerV1().run(
            permit=permit,
            binding=binding,
            gate=gate,
            pre_execution=pre_execution,
        )
        raw_path = f"artifacts/{binding.run_id}/raw_run_output.v1.json"
        raw_artifact = self._store.register_artifact(
            RegisterArtifactCommand(
                round_id=str(binding.round_id),
                artifact_type="RAW_RUN_OUTPUT_V1",
                relative_path=raw_path,
                producer="FakeNonTrainingRunnerV1",
                idempotency_key=f"raw-output:{claim['claim_id']}",
            ),
            canonical_json_bytes(raw_output.to_dict()),
        )
        self._store.mark_execution_finished(
            MarkExecutionFinishedCommand(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                raw_output_artifact_id=str(raw_artifact["artifact_id"]),
                idempotency_key=f"execution-finish:{claim['claim_id']}",
            )
        )
        return receipt, raw_output, (receipt_artifact, raw_artifact)


def register_raw_result_envelope(
    store: SingleWriterExperimentStoreV1,
    envelope: RawResultEnvelopeV1,
) -> dict[str, Any]:
    return store.register_artifact(
        RegisterArtifactCommand(
            round_id=str(envelope.round_id),
            artifact_type="RAW_RESULT_ENVELOPE_V1",
            relative_path=f"artifacts/{envelope.run_id}/raw_result_envelope.v1.json",
            producer="RawResultEnvelopeWriterV1",
            idempotency_key=f"raw-result-envelope:{envelope.round_id}",
        ),
        canonical_json_bytes(envelope.to_dict()),
    )


__all__ = [
    "FakeNonTrainingRunnerV1",
    "PackageOwnedLauncherV1",
    "register_raw_result_envelope",
]
