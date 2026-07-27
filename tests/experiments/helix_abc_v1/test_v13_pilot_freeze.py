from __future__ import annotations

import hashlib
import json
import runpy
import sqlite3
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v13 import (
    V13_EXECUTABLE_PROFILE_DIGEST,
    V13_PILOT_ROUNDS_PER_ARM,
    V13_PILOT_SEARCH_SEED,
    V13PilotOrchestratorV1,
    V13PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ControllerKind,
    EvidencePortKind,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    POLICY_BUNDLE_DIGEST_V18,
    MetaV18CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_COMMIT,
    PinnedOriginalMainAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (
    RealPilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    CAMPAIGN_TRAINING_RUNNER_ABI,
    campaign_training_runtime_release,
    validate_campaign_training_runtime_release,
)
from recclaw_core.helix.guard_adapter import EvidenceGuardPortV1
from recclaw_core.helix.ports import NullEvidencePortV1


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/recclaw_core/experiments/helix_abc_v1"
TRAINING_PYTHON = Path(
    "/root/projects/RecClaw_m6_training_runtime_v2/bin/python"
)
RECBOLE_ROOT = Path("/root/projects/RecBole_m6_runtime")
SEARCH_DATASET = Path("/root/projects/RecClaw_campaign_dataset_v1/search")


class _NoCallUpstream:
    model = "no-call-fixture"
    max_total_tokens_per_call = 20_000

    def __init__(self) -> None:
        self.calls = 0

    def call_with_session(self, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("Pilot construction must not call the provider")


def test_v13_three_arm_policy_and_main_grade_profile_are_exact() -> None:
    contract = V13PilotStoreContractV1.create()
    profile = campaign_runtime_profile()
    assert contract.search_seeds == (V13_PILOT_SEARCH_SEED,)
    assert contract.scheduled_slots_per_arm_seed == V13_PILOT_ROUNDS_PER_ARM
    assert profile["profile_id"] == "BL_ICF_EXECUTABLE_PROFILE_V2"
    assert profile["executable_profile_digest"] == (
        V13_EXECUTABLE_PROFILE_DIGEST
    )
    assert len(executable_mechanisms()) == 66
    assert {
        item.bl_icf_search_space_digest for item in contract.arm_policies
    } == {V13_EXECUTABLE_PROFILE_DIGEST}
    by_arm = {item.arm: item for item in contract.arm_policies}
    assert by_arm[ArmCode.B].non_guard_projection() == (
        by_arm[ArmCode.C].non_guard_projection()
    )
    assert by_arm[ArmCode.A].controller is ControllerKind.ORIGINAL
    assert by_arm[ArmCode.B].evidence_port is EvidencePortKind.NULL
    assert by_arm[ArmCode.C].evidence_port is EvidencePortKind.EVIDENCE_GUARD


def test_v13_broker_schema_and_training_release_bind_current_runtime() -> None:
    broker_release_path = (
        PACKAGE / "resources/lab_api_broker_release_v1_v13_schema_v4.json"
    )
    schema_path = (
        PACKAGE / "resources/campaign_proposal_response_v1.schema.json"
    )
    broker_release = json.loads(
        broker_release_path.read_text(encoding="utf-8")
    )
    assert broker_release["model"] == "gpt-5.4"
    assert broker_release["response_schema_digest"] == (
        hashlib.sha256(schema_path.read_bytes()).hexdigest()
    )
    release = campaign_training_runtime_release()
    assert release.release_id == "TRAINING_RUNTIME_RELEASE_V5"
    assert release.runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
    assert validate_campaign_training_runtime_release(
        data_path=SEARCH_DATASET,
        python_executable=TRAINING_PYTHON,
        recbole_root=RECBOLE_ROOT,
    ) == ()


def test_v13_construction_uses_pinned_main_v18_and_real_training_hooks(
    tmp_path: Path,
) -> None:
    upstream = _NoCallUpstream()
    runtime = MetaV18CampaignRuntimeV1(
        checkpoint_path=(
            PACKAGE / "resources/meta_vnext_policy_checkpoint_v18.json"
        ),
        experiment_id=V13PilotStoreContractV1.create().experiment_id,
        search_seed=V13_PILOT_SEARCH_SEED,
        scheduled_rounds=V13_PILOT_ROUNDS_PER_ARM,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    broker = RealCanaryProposalBrokerV1.create_v13(
        upstream=upstream,
        template_path=PACKAGE / "resources/campaign_anchor_programs_v1.json",
        repository_root=ROOT,
        search_seed=V13_PILOT_SEARCH_SEED,
        adaptive_memory=True,
        campaign_meta_runtime=runtime,
    )
    assert broker.v13_mode is True
    assert isinstance(broker.original_controller, PinnedOriginalMainAdapterV1)
    assert broker.original_controller._source_release.repository_root == ROOT
    assert broker.original_controller.source_release_digest
    assert ORIGINAL_MAIN_COMMIT == (
        "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
    )
    assert V13PilotOrchestratorV1._execute_selected is (
        RealPilotOrchestratorV1._execute_selected
    )
    with V13PilotOrchestratorV1(
        tmp_path / "runtime",
        broker=broker,
        meta_runtime=runtime,
        project_root=ROOT,
        recbole_root=RECBOLE_ROOT,
        data_path=SEARCH_DATASET,
        python_executable=TRAINING_PYTHON,
    ) as orchestrator:
        assert isinstance(orchestrator.ports[ArmCode.B], NullEvidencePortV1)
        assert isinstance(orchestrator.ports[ArmCode.C], EvidenceGuardPortV1)
        assert runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V18
        assert set(runtime._states) == {ArmCode.B, ArmCode.C}
        assert orchestrator.store._connection.execute(
            "SELECT COUNT(*) FROM rounds"
        ).fetchone()[0] == 0
    assert upstream.calls == 0


def test_v13_freeze_build_and_entrypoint_import_have_no_run_side_effect() -> None:
    from scripts.freeze_v13_pilot_contract import (
        DEFAULT_LLM_CONFIG,
        DEFAULT_OUTPUT_ROOT,
        build_contract,
    )

    assert not DEFAULT_OUTPUT_ROOT.exists()
    contract = build_contract(llm_api_config=DEFAULT_LLM_CONFIG)
    assert contract["pilot_started"] is False
    assert contract["original"]["adapter"] == "PinnedOriginalMainAdapterV1"
    assert contract["original"]["main_commit"] == ORIGINAL_MAIN_COMMIT
    assert contract["broker"]["model"] == "gpt-5.4"
    assert contract["common_substrate"][
        "pilot_main_profile_identity_equal"
    ]
    assert contract["meta"]["policy_bundle_digest"] == (
        POLICY_BUNDLE_DIGEST_V18
    )
    assert contract["training"]["release_id"] == (
        "TRAINING_RUNTIME_RELEASE_V5"
    )
    runpy.run_path(str(ROOT / "scripts/run_v13_pilot.py"))
    assert not DEFAULT_OUTPUT_ROOT.exists()


def test_v13_analysis_reads_training_v2_result_and_feedback_event(
    tmp_path: Path,
) -> None:
    from scripts.run_v13_pilot import _collect_analysis_rows

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    relative_path = Path("raw/result.json")
    (artifact_root / relative_path).parent.mkdir()
    (artifact_root / relative_path).write_text(
        json.dumps(
            {
                "candidate_id": "bl1_v13_analysis",
                "exit_status": "SUCCESS",
                "normalized_metrics": {"ndcg": 0.123},
                "seed": 2026,
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE rounds (
                round_id TEXT PRIMARY KEY,
                arm_instance_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                terminal_class TEXT,
                status TEXT NOT NULL
            );
            CREATE TABLE artifact_index (
                round_id TEXT,
                artifact_type TEXT NOT NULL,
                relative_path TEXT NOT NULL
            );
            CREATE TABLE round_events (
                round_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE resource_ledger (
                round_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                quantity INTEGER NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO rounds VALUES (?, ?, ?, ?, ?)",
            ("round-1", "opaque-a", 1, "COMPLETED", "CLOSED"),
        )
        connection.execute(
            "INSERT INTO artifact_index VALUES (?, ?, ?)",
            (
                "round-1",
                "RAW_RESULT_ENVELOPE_V2",
                relative_path.as_posix(),
            ),
        )
        connection.execute(
            "INSERT INTO round_events VALUES (?, ?, ?)",
            (
                "round-1",
                "ROUND_FEEDBACK",
                json.dumps(
                    {
                        "feedback": {
                            "frontier_eligibility": (
                                "ELIGIBLE_DEVELOPMENT_FRONTIER"
                            )
                        }
                    }
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO resource_ledger VALUES (?, ?, ?)",
            (
                ("round-1", "ORDINARY_EXECUTION", 1),
                ("round-1", "BILLED_TOKEN_DEBIT", 10),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    rows, eligibility = _collect_analysis_rows(db_path, artifact_root)
    assert rows == [
        {
            **rows[0],
            "billed_tokens": 10,
            "candidate_id": "bl1_v13_analysis",
            "ndcg": 0.123,
            "observation_seed": "2026",
            "ordinary_execution_count": 1,
            "run_status": "SUCCESS",
        }
    ]
    assert tuple(eligibility.values()) == (
        "ELIGIBLE_DEVELOPMENT_FRONTIER",
    )
