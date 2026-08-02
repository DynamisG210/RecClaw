from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
)
from recclaw_core.experiments.helix_abc_v1.multiround_soak import (  # noqa: E402
    MultiRoundSoakError,
    create_stage_ledger,
    freeze_round_manifest,
    resume_round,
    seal_stage,
)


Q3_ROOT = ROOT / "results/research_line/q3_outcome_aware_learning_20260803_01"
POLICY = Q3_ROOT / "policy/versioned_policy.json"
ACTIVATION = Q3_ROOT / "activation/active_policy.json"
POOL = (
    ROOT
    / "results/research_line/q1_prompt_contract_20260802_01/"
    "FROZEN_SELECTION_BEFORE_IMPLEMENTATION.json"
)


def _frozen_execution() -> dict[str, object]:
    return {
        "provider_endpoint_digest": "1" * 64,
        "provider_model_digest": "2" * 64,
        "data_digest": "3" * 64,
        "config_digest": "4" * 64,
        "budget_seconds": 7200,
        "deadline_seconds": 900,
        "denominator_rule_digest": "5" * 64,
        "outcome_interpretation_digest": "6" * 64,
        "held_out_reads": 0,
    }


def _freeze(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "ROUND_MANIFEST.json"
    freeze_round_manifest(
        output_path=manifest_path,
        campaign_id="q4-multiround-soak-test",
        round_index=1,
        policy_path=POLICY,
        activation_path=ACTIVATION,
        full_pool_path=POOL,
        acquisition_seeds={"IDEA": 41001, "EXPERIMENT": 41002, "REPLICATION": 41003},
        execution_task_type="EXPERIMENT",
        frozen_execution=_frozen_execution(),
        previous_round_activation_digest=None,
    )
    return manifest_path


def test_freeze_manifest_consumes_disk_policy_and_complete_pool(
    tmp_path: Path,
) -> None:
    manifest_path = _freeze(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["campaign_id"] == "q4-multiround-soak-test"
    assert manifest["round_index"] == 1
    assert manifest["stage"] == "MANIFEST_FROZEN"
    assert manifest["execution_task_type"] == "EXPERIMENT"
    assert manifest["held_out_reads"] == 0
    assert manifest["frozen_inputs"]["policy_file_sha256"] == bytes_sha256(
        POLICY.read_bytes()
    )
    assert manifest["frozen_inputs"]["activation_file_sha256"] == bytes_sha256(
        ACTIVATION.read_bytes()
    )
    assert manifest["frozen_inputs"]["full_pool_file_sha256"] == bytes_sha256(
        POOL.read_bytes()
    )
    assert set(manifest["task_acquisitions"]) == {
        "IDEA",
        "EXPERIMENT",
        "REPLICATION",
    }
    for acquisition in manifest["task_acquisitions"].values():
        assert acquisition["consumer_input"]["read_active_policy_from_disk"] is True
        assert acquisition["candidate_count"] == len(acquisition["candidates"]) == 4
        assert acquisition["exploration_probability"] == 0.15
        assert acquisition["selection_probabilities_sum"] == 1.0
        assert sum(row["selected"] for row in acquisition["candidates"]) == 1
        assert all("head_predictions" in row for row in acquisition["candidates"])
        assert all("selection_score" in row for row in acquisition["candidates"])
        assert all("selection_probability" in row for row in acquisition["candidates"])


def test_resume_skips_sealed_stage_and_preserves_physical_call_identity(
    tmp_path: Path,
) -> None:
    manifest_path = _freeze(tmp_path)
    ledger_path = tmp_path / "STAGE_LEDGER.json"
    create_stage_ledger(manifest_path=manifest_path, ledger_path=ledger_path)

    provider_artifact = tmp_path / "provider-result.json"
    provider_artifact.write_text('{"status":"RESOLVED"}\n', encoding="utf-8")
    sealed = seal_stage(
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        stage="PROVIDER_RESOLVER",
        status="SEALED_SUCCESS",
        artifact_paths=(provider_artifact,),
        process={
            "host": "gpu35",
            "logical_call_id": "round-1-provider",
            "physical_call_ids": ["round-1-provider-physical-1"],
            "return_code": 0,
        },
    )
    assert sealed["process"]["physical_call_ids"] == [
        "round-1-provider-physical-1"
    ]

    resumed = resume_round(manifest_path=manifest_path, ledger_path=ledger_path)
    assert resumed["next_stage"] == "IMPLEMENTER"
    assert resumed["sealed_stages"] == ["MANIFEST_FROZEN", "PROVIDER_RESOLVER"]
    assert resumed["reuse_forbidden_physical_call_ids"] == [
        "round-1-provider-physical-1"
    ]

    with pytest.raises(MultiRoundSoakError, match="already sealed"):
        seal_stage(
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            stage="PROVIDER_RESOLVER",
            status="SEALED_SUCCESS",
            artifact_paths=(provider_artifact,),
            process={
                "host": "gpu35",
                "logical_call_id": "round-1-provider",
                "physical_call_ids": ["round-1-provider-physical-2"],
                "return_code": 0,
            },
        )


def test_stage_order_and_sealed_artifact_hash_are_fail_closed(tmp_path: Path) -> None:
    manifest_path = _freeze(tmp_path)
    ledger_path = tmp_path / "STAGE_LEDGER.json"
    create_stage_ledger(manifest_path=manifest_path, ledger_path=ledger_path)
    artifact = tmp_path / "implementation.json"
    artifact.write_text('{"status":"PASS"}\n', encoding="utf-8")

    with pytest.raises(MultiRoundSoakError, match="next stage"):
        seal_stage(
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            stage="IMPLEMENTER",
            status="SEALED_SUCCESS",
            artifact_paths=(artifact,),
            process={"host": "gpu35", "physical_call_ids": [], "return_code": 0},
        )

    provider_artifact = tmp_path / "provider.json"
    provider_artifact.write_text('{"status":"PASS"}\n', encoding="utf-8")
    seal_stage(
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        stage="PROVIDER_RESOLVER",
        status="SEALED_SUCCESS",
        artifact_paths=(provider_artifact,),
        process={"host": "gpu35", "physical_call_ids": [], "return_code": 0},
    )
    provider_artifact.write_text('{"status":"DRIFTED"}\n', encoding="utf-8")

    with pytest.raises(MultiRoundSoakError, match="artifact hash drift"):
        resume_round(manifest_path=manifest_path, ledger_path=ledger_path)
