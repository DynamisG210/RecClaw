from __future__ import annotations

from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v15 import (
    V15PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_COMMIT,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    protected_side_effect_manifest,
    side_effect_audit,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    campaign_training_runtime_release,
)


def test_v15_preserves_exact_three_arm_attribution() -> None:
    contract = V15PilotStoreContractV1.create()
    by_arm = {item.arm: item for item in contract.arm_policies}
    assert by_arm[ArmCode.B].non_guard_projection() == (
        by_arm[ArmCode.C].non_guard_projection()
    )
    assert by_arm[ArmCode.B].evidence_port.value == "NullEvidencePortV1"
    assert by_arm[ArmCode.C].evidence_port.value == "EvidenceGuardPortV1"
    assert ORIGINAL_MAIN_COMMIT == (
        "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
    )


def test_v8_release_binds_native_backend_and_hash_confinement() -> None:
    release = campaign_training_runtime_release()
    assert release.release_id == "TRAINING_RUNTIME_RELEASE_V8"
    assert release.backend_identity["backend_class"] == (
        "LAB_GPU5_RTX4090_NATIVE_LINUX_V1"
    )
    assert release.backend_identity["filesystem_mode"] == (
        "HASH_AUDITED_PRIVATE_ROOT_V1"
    )
    assert release.backend_identity["torch_cuda_environment"][
        "primary_device_name"
    ] == "NVIDIA GeForce RTX 4090"
    assert release.backend_identity["python_package_versions"]["scipy"] == (
        "1.12.0"
    )


def test_sibling_arm_write_is_detected(tmp_path: Path) -> None:
    arm_b = tmp_path / "instances" / "arm-b"
    arm_c = tmp_path / "instances" / "arm-c"
    arm_b.mkdir(parents=True)
    arm_c.mkdir(parents=True)
    (arm_b / "state.json").write_text("before", encoding="utf-8")
    before = protected_side_effect_manifest({"sibling_arm:b": arm_b})
    (arm_c / "own.json").write_text("allowed", encoding="utf-8")
    unchanged = protected_side_effect_manifest({"sibling_arm:b": arm_b})
    assert side_effect_audit(before, unchanged)["status"] == "PASS"
    (arm_b / "state.json").write_text("mutated", encoding="utf-8")
    changed = protected_side_effect_manifest({"sibling_arm:b": arm_b})
    assert side_effect_audit(before, changed)["status"] == "FAIL"
