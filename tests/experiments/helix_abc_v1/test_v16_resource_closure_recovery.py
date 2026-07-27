from __future__ import annotations

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
    V16_PILOT_ROUNDS_PER_ARM,
    V16_PILOT_SEARCH_SEED,
    V16PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    campaign_training_runtime_release,
    resolve_bound_training_release,
    resolve_runtime_release,
)


def test_v16_is_fresh_and_preserves_exact_three_arm_attribution() -> None:
    contract = V16PilotStoreContractV1.create()
    by_arm = {item.arm: item for item in contract.arm_policies}
    assert V16_PILOT_SEARCH_SEED == 9218
    assert V16_PILOT_ROUNDS_PER_ARM == 5
    assert by_arm[ArmCode.B].non_guard_projection() == (
        by_arm[ArmCode.C].non_guard_projection()
    )
    assert by_arm[ArmCode.B].evidence_port.value == "NullEvidencePortV1"
    assert by_arm[ArmCode.C].evidence_port.value == "EvidenceGuardPortV1"


def test_v16_binds_training_runtime_v9_end_to_end() -> None:
    release = campaign_training_runtime_release()
    assert release.release_id == "TRAINING_RUNTIME_RELEASE_V9"
    resolved = resolve_runtime_release(release.runner_abi)
    bound = resolve_bound_training_release(
        runner_abi=release.runner_abi,
        runtime_release_digest=release.digest,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
    )
    assert resolved["release_digest"] == release.digest
    assert bound.digest == release.digest
