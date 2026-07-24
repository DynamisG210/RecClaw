"""Additive candidate/runtime binding for package-owned training."""

from __future__ import annotations

from pathlib import Path

from .canonical import sha256_digest
from .runtime_contracts import CandidateExecutionBindingV2
from .training_runtime_contracts import (
    CandidateExecutionBindingV3,
    TrainingExecutionPurposeV1,
    TrainingRuntimeBindingV2,
)
from .training_runtime_release import TRAINING_RUNNER_ABI


def build_training_binding_v3(
    *,
    base_binding: CandidateExecutionBindingV2,
    runtime_binding: TrainingRuntimeBindingV2,
) -> CandidateExecutionBindingV3:
    if runtime_binding.execution_purpose not in {
        item.value for item in TrainingExecutionPurposeV1
    }:
        raise ValueError("training runtime binding has an unknown purpose")
    expected = {
        "budget_digest": base_binding.budget_digest,
        "candidate_id": base_binding.candidate_id,
        "implementation_digest": base_binding.implementation_digest,
        "opaque_arm_instance_id": base_binding.opaque_arm_instance_id,
        "round_id": base_binding.round_id,
        "run_id": base_binding.run_id,
    }
    for field, value in expected.items():
        if getattr(runtime_binding, field) != value:
            raise ValueError(f"training runtime binding mismatch: {field}")
    if runtime_binding.runner_abi != TRAINING_RUNNER_ABI:
        raise ValueError("training runtime binding uses the wrong ABI")
    if runtime_binding.instance_private_root_digest != sha256_digest(
        {
            "resolved_instance_private_root": Path(
                str(base_binding.arm_private_root)
            ).resolve().as_posix()
        }
    ):
        raise ValueError("training runtime binding uses another private root")
    return CandidateExecutionBindingV3(
        {
            "arm_private_root": base_binding.arm_private_root,
            "base_binding_digest": base_binding.digest,
            "budget_digest": base_binding.budget_digest,
            "candidate_id": base_binding.candidate_id,
            "execution_purpose": runtime_binding.execution_purpose,
            "implementation_digest": base_binding.implementation_digest,
            "mechanism_program_digest": base_binding.mechanism_program_digest,
            "mechanism_semantics_digest": base_binding.mechanism_semantics_digest,
            "opaque_arm_instance_id": base_binding.opaque_arm_instance_id,
            "profile_digest": base_binding.profile_digest,
            "protocol_digest": runtime_binding.protocol_digest,
            "round_id": base_binding.round_id,
            "run_id": base_binding.run_id,
            "runner_abi": runtime_binding.runner_abi,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": runtime_binding.release_digest,
            "search_seed": base_binding.search_seed,
        }
    )


def verify_training_binding_v3(
    binding: CandidateExecutionBindingV3,
    *,
    base_binding: CandidateExecutionBindingV2,
    runtime_binding: TrainingRuntimeBindingV2,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        expected = build_training_binding_v3(
            base_binding=base_binding,
            runtime_binding=runtime_binding,
        )
    except ValueError as error:
        reasons.append(str(error))
    else:
        if binding.to_dict() != expected.to_dict():
            reasons.append("TRAINING_CANDIDATE_BINDING_SUBSTITUTION")
    return not reasons, tuple(sorted(reasons))


__all__ = ["build_training_binding_v3", "verify_training_binding_v3"]
