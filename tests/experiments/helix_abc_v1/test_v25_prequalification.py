from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import sys
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
    v16_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v25 import (
    V25_EXECUTABLE_PROFILE_DIGEST,
    V25_PILOT_CHECKPOINTS,
    V25_PILOT_ROUNDS_PER_ARM,
    V25_PILOT_SEARCH_SEED,
    V25PilotOrchestratorV1,
    v25_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    MetaV20CampaignRuntimeV1,
    meta_v20_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (
    MetaV17PilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.producer_opportunity import (
    PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (
    TrainingRuntimeReleaseV3,
)


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_v13_pilot import _collect_analysis_rows  # noqa: E402


RESOURCE_ROOT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
)


def test_v25_preserves_treatment_and_binds_fresh_effect_identity() -> None:
    profile = campaign_runtime_profile()
    assert tuple(item.to_dict() for item in v25_arm_policies()) == tuple(
        item.to_dict() for item in v16_arm_policies()
    )
    assert V25_PILOT_SEARCH_SEED == 9227
    assert V25_PILOT_ROUNDS_PER_ARM == 50
    assert V25_PILOT_CHECKPOINTS == (10, 20, 50)
    assert profile["profile_id"] == "BL_ICF_EXECUTABLE_PROFILE_V2"
    assert profile["executable_mechanism_count"] == 66
    assert (
        profile["executable_profile_digest"]
        == V25_EXECUTABLE_PROFILE_DIGEST
    )


def test_v25_runtime_binds_meta_v20_and_resource_envelope() -> None:
    assert issubclass(V25PilotOrchestratorV1, MetaV17PilotOrchestratorV1)
    source = inspect.getsource(V25PilotOrchestratorV1)
    assert "MetaV20CampaignRuntimeV1" in source
    assert "_resource_ceilings=expected_ceilings" in source
    assert "v25_resource_ceilings()" in source
    assert "pilot_budget =" not in source
    builder = (ROOT / "scripts/build_v25_gpu35_closure.py").read_text(
        encoding="utf-8"
    )
    independent_audit = (
        ROOT / "scripts/audit_v25_m6i_prelaunch.py"
    ).read_text(encoding="utf-8")
    effect_audit = (
        ROOT / "scripts/analyze_v25_effect_pilot.py"
    ).read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_v25_effect_pilot.py").read_text(
        encoding="utf-8"
    )
    analysis_rows = (ROOT / "scripts/run_v13_pilot.py").read_text(
        encoding="utf-8"
    )
    assert "M6I_V25_FINAL_INDEPENDENT_AUDIT.json" in builder
    assert "M6I V25 independent audit is not PASS" in builder
    assert "V25_GPU35_RESOURCE_MARGIN_POLICY_V2.json" in builder
    assert "V25_COMPOSITIONAL_CANARY_9385_FAILURE.json" in builder
    assert "compositional_recovery" in builder
    assert "CANARY_RECOVERY_SOURCE_HEAD" in builder
    assert "LabApiCanaryBrokerV1" not in independent_audit
    assert "campaign_train_worker" not in independent_audit
    assert "effect_pilot_verdict" in effect_audit
    assert "seed_binding_mismatch_count" in effect_audit
    assert "report = analyze(contract_path)" in runner
    assert '"mechanism_semantics_digest"' in analysis_rows


def test_v25_checkpoints_are_barrier_complete_and_read_only() -> None:
    source = inspect.getsource(
        V25PilotOrchestratorV1._persist_read_only_checkpoint
    )
    assert "closed_triplet_barriers" in source
    assert "open_rounds" in source
    assert "EFFECT_PILOT_READ_ONLY_CHECKPOINT_V1" in source
    assert "register_artifact" in source
    assert "update" not in source.lower()
    assert "delete" not in source.lower()
    safety_source = inspect.getsource(
        V25PilotOrchestratorV1.immutable_audit_bundle
    )
    assert "m6i_safety_projection" in safety_source
    assert "cross_arm_physical_identities" in safety_source
    assert "guard_private_context_token_count" in safety_source


def test_v25_meta_checkpoint_is_exact_and_not_a_promotion() -> None:
    path = RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v20.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5ff049e563ac73a192ad657cdabcf11c014d58a55d95838e9bbb63a1c2b11227"
    )
    assert checkpoint["formal_acceptance"] is False
    assert checkpoint["activation_boundary"] == "NEXT_FRESH_CAMPAIGN"
    assert checkpoint["coefficient_action"] == (
        "INHERIT_EXACT_V19_NO_COEFFICIENT_CHANGE"
    )
    assert (
        checkpoint["producer_opportunity_policy"]["policy_digest"]
        == PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
    )
    assert checkpoint["policy_bundle_digest"] == (
        meta_v20_research_control_policy().meta_router_policy_digest
    )
    assert checkpoint["parent_policy_bundle_digest"] == (
        "95923b800e89c4c4bfb994b5aa8a16069ac8429af056b456f01b42af4ab33744"
    )


def test_v25_runtime_release_v17_has_closed_digest() -> None:
    path = RESOURCE_ROOT / "training_runtime_release_v17.json"
    release = json.loads(path.read_text(encoding="utf-8"))
    assert release["release_id"] == "TRAINING_RUNTIME_RELEASE_V17"
    expected = TrainingRuntimeReleaseV3(release).digest
    assert expected == (
        "eec5a5a7482e56c7f5ed7b899d60236539346394d2ef6c828f9c2ee40f27bc9d"
    )


def test_v25_canary_recovery_is_versioned_and_preserves_resource_rule() -> None:
    docs = ROOT / "docs/research_line/continuous_program"
    v1 = json.loads(
        (docs / "V25_GPU35_RESOURCE_MARGIN_POLICY_V1.json").read_text(
            encoding="utf-8"
        )
    )
    v2 = json.loads(
        (docs / "V25_GPU35_RESOURCE_MARGIN_POLICY_V2.json").read_text(
            encoding="utf-8"
        )
    )
    failure = json.loads(
        (docs / "V25_COMPOSITIONAL_CANARY_9385_FAILURE.json").read_text(
            encoding="utf-8"
        )
    )
    launcher = (
        ROOT / "scripts/launch_v24_qualification_canary.py"
    ).read_text(encoding="utf-8")
    assert v2["margin_policy"] == v1["margin_policy"]
    assert v2["predeclared_qualification_ceiling"] == (
        v1["predeclared_qualification_ceiling"]
    )
    assert v2["qualification_process_environment"][
        "PYTHONDONTWRITEBYTECODE"
    ] == "1"
    assert [row["search_seed"] for row in v2["canary_specs"]] == [
        9383,
        9384,
        9387,
        9386,
    ]
    assert failure["failed_root_policy"]["reuse_allowed"] is False
    assert failure["observations"]["resource_ceiling_status"] == "VERIFIED"
    assert failure["scientific_interpretation"][
        "candidate_effect_evidence"
    ] is False
    assert '"PYTHONDONTWRITEBYTECODE"] = "1"' in launcher
    assert '"source_bytecode_write_disabled"' in launcher


def test_v25_analysis_rows_bind_executed_semantics(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    connection = sqlite3.connect(state)
    connection.executescript(
        """
        CREATE TABLE rounds (
            round_id TEXT,
            arm_instance_id TEXT,
            round_index INTEGER,
            terminal_class TEXT,
            status TEXT
        );
        CREATE TABLE artifact_index (
            round_id TEXT,
            artifact_type TEXT,
            relative_path TEXT
        );
        CREATE TABLE round_events (
            round_id TEXT,
            event_type TEXT,
            payload_json TEXT
        );
        CREATE TABLE resource_ledger (
            round_id TEXT,
            dimension TEXT,
            quantity INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO rounds VALUES ('r1','opaque-a',1,'COMPLETED','CLOSED')"
    )
    connection.execute(
        "INSERT INTO artifact_index VALUES "
        "('r1','RAW_RESULT_ENVELOPE_V2','raw.json')"
    )
    feedback = {
        "feedback": {
            "candidate_id": "candidate-1",
            "frontier_eligibility": "SEARCH_ELIGIBLE",
            "search_utility_event": {
                "candidate_semantic_digest": "a" * 64,
            },
        }
    }
    connection.execute(
        "INSERT INTO round_events VALUES (?,?,?)",
        ("r1", "ROUND_FEEDBACK", json.dumps(feedback)),
    )
    connection.executemany(
        "INSERT INTO resource_ledger VALUES (?,?,?)",
        [
            ("r1", "BILLED_TOKEN_DEBIT", 10),
            ("r1", "GPU_COST_MICROUNITS", 20),
            ("r1", "ORDINARY_EXECUTION", 1),
        ],
    )
    connection.commit()
    connection.close()
    (artifacts / "raw.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-1",
                "exit_status": "SUCCESS",
                "normalized_metrics": {"ndcg": 0.2},
                "seed": 2026,
            }
        ),
        encoding="utf-8",
    )
    rows, eligibility = _collect_analysis_rows(state, artifacts)
    assert rows[0]["mechanism_semantics_digest"] == "a" * 64
    assert rows[0]["candidate_id"] == "candidate-1"
    assert list(eligibility.values()) == ["SEARCH_ELIGIBLE"]
