from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1 import (
    SingleWriterExperimentStoreV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v15 import (
    V15PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_COMMIT,
    ORIGINAL_MAIN_FILES,
    OriginalMainSourceReleaseV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    _round_execution_budget_debits,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import pilot_budget
from recclaw_core.experiments.helix_abc_v1.state_store import (
    CloseRoundCommand,
    OpenRoundCommand,
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


def test_pinned_original_materializes_exact_main_blobs(tmp_path: Path) -> None:
    release = OriginalMainSourceReleaseV1(
        repository_root=Path(__file__).resolve().parents[3],
        materialization_root=tmp_path / "original",
    )
    release.materialize()
    for relative, (_blob_sha1, expected_sha256) in ORIGINAL_MAIN_FILES.items():
        observed = hashlib.sha256(
            (tmp_path / "original" / relative).read_bytes()
        ).hexdigest()
        assert observed == expected_sha256


def test_resource_rejection_preserves_actual_but_debits_allocation() -> None:
    ceilings = pilot_budget()
    actual = {
        "gpu_device_time_ms": 1_368_131,
        "gpu_cost_microunits": 380_036,
        "wall_time_ms": 1_368_131,
    }
    debits = {
        item.dimension: item.quantity
        for item in _round_execution_budget_debits(
            **actual,
            ceilings=ceilings,
            resource_ceiling_rejected=True,
        )
    }
    assert actual["gpu_device_time_ms"] == 1_368_131
    assert actual["gpu_cost_microunits"] == 380_036
    assert debits == {
        "GPU_COST_MICROUNITS": ceilings.gpu_cost_microunits,
        "GPU_DEVICE_TIME_MS": ceilings.gpu_device_time_ms,
        "WALL_TIME_MS": actual["wall_time_ms"],
    }


def test_resource_rejection_closes_round_with_capped_allocation_debit(
    tmp_path: Path,
) -> None:
    ceilings = pilot_budget()
    actual = {
        "gpu_device_time_ms": 1_368_131,
        "gpu_cost_microunits": 380_036,
        "wall_time_ms": 1_368_131,
    }
    contract = default_experiment_contract()
    genesis = sha256_digest(
        {
            "experiment_contract_digest": contract.identity_digest,
            "state": "GENESIS",
        }
    )
    with SingleWriterExperimentStoreV1(
        tmp_path / "state.sqlite3",
        tmp_path / "artifacts",
    ) as store:
        arm_ids = store.initialize_experiment(contract)
        opened = store.open_round(
            OpenRoundCommand(
                experiment_id=contract.experiment_id,
                arm_instance_id=arm_ids[ArmCode.A],
                arm_code=ArmCode.A,
                search_seed=42,
                round_index=1,
                budget_snapshot=ceilings,
                controller_state_before_digest=genesis,
                idempotency_key="open:resource-rejection",
            )
        )
        closed = store.close_round(
            CloseRoundCommand(
                round_id=opened["round_id"],
                terminal_class="COMPLETED",
                feedback_payload={
                    "outcome_class": "TRAINING_RESULT_REJECTED",
                    "reason_code": "TRAINING_RESOURCE_CEILING_EXCEEDED",
                    "actual_resource_audit": actual,
                },
                controller_state_after_digest=sha256_digest(
                    {"round_id": opened["round_id"], "state": "AFTER"}
                ),
                resource_debits=_round_execution_budget_debits(
                    **actual,
                    ceilings=ceilings,
                    resource_ceiling_rejected=True,
                ),
                idempotency_key="close:resource-rejection",
            )
        )
        assert closed["status"] == "CLOSED"

    connection = sqlite3.connect(tmp_path / "state.sqlite3")
    ledger = dict(
        connection.execute(
            """
            SELECT dimension, SUM(quantity)
            FROM resource_ledger
            WHERE dimension IN (
                'GPU_DEVICE_TIME_MS',
                'GPU_COST_MICROUNITS',
                'WALL_TIME_MS'
            )
            GROUP BY dimension
            """
        ).fetchall()
    )
    round_row = connection.execute(
        "SELECT status, terminal_class FROM rounds WHERE round_id = ?",
        (opened["round_id"],),
    ).fetchone()
    feedback_count = connection.execute(
        """
        SELECT COUNT(*) FROM round_events
        WHERE round_id = ? AND event_type = 'ROUND_FEEDBACK'
        """,
        (opened["round_id"],),
    ).fetchone()[0]
    connection.close()

    assert ledger == {
        "GPU_COST_MICROUNITS": ceilings.gpu_cost_microunits,
        "GPU_DEVICE_TIME_MS": ceilings.gpu_device_time_ms,
        "WALL_TIME_MS": actual["wall_time_ms"],
    }
    assert round_row == ("CLOSED", "COMPLETED")
    assert feedback_count == 1
    assert actual["gpu_device_time_ms"] > ledger["GPU_DEVICE_TIME_MS"]
