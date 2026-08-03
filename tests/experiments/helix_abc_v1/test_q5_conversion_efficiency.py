from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (  # noqa: E402
    CANDIDATE_LOCAL_ALLOWED_FILES,
    build_conversion_execution_plan,
    build_mechanical_repair_request,
    build_stage_feasibility_head,
    choose_stable_promotions,
    finalize_conversion_execution_plan,
    is_mechanical_repair_failure,
    run_fail_soft_batch,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (  # noqa: E402
    InnovationSpineError,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
)


def _digest(label: str) -> str:
    return (label.encode("utf-8").hex() + "0" * 64)[:64]


def _spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="A candidate-local module improves propagation control.",
        mechanism_change="Add a candidate-local helper module.",
        competing_explanation="The change may be inert.",
        matched_control_requirement="Keep the parent model and data fixed.",
        implementation_requirements=("Implement a RecBole recommender.",),
        expected_evidence=("Construction and API checks.",),
        falsifier="Reject if the frozen interface cannot be satisfied.",
        compatibility_requirements=("RecBole GeneralRecommender.",),
        protocol_ref="protocol:conversion-fixture",
        protocol_digest=_digest("protocol"),
        context_ref="context:conversion-fixture",
        context_digest=_digest("context"),
        current_profile_ref="profile:conversion-fixture",
        current_profile_digest=_digest("profile"),
        producer_role="mechanism_composer",
        high_change_justification="The module is outside the frozen baseline catalog.",
        current_profile_expressibility_claim=CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE,
    )


def test_mechanical_repair_is_blind_and_qualification_or_effect_failure_is_not() -> None:
    original = {"schema": "request", "blind_research_spec": {"hypothesis": "x"}}
    failure = {
        "stage": "API_CONTRACT",
        "failure_class": "INTERFACE",
        "reason_code": "PREDICT_SIGNATURE_INVALID",
    }
    assert is_mechanical_repair_failure(failure)
    repaired = build_mechanical_repair_request(
        original,
        failure,
        current_source={"recclaw_ext/candidate.py": "class CandidateModel: pass\n"},
        failure_message="predict signature mismatch",
        short_trace="API_CONTRACT/PREDICT_SIGNATURE_INVALID",
    )
    assert repaired["repair_context"]["stage"] == "API_CONTRACT"
    assert repaired["repair_context"]["current_source_files"]["recclaw_ext/candidate.py"]
    assert repaired["repair_attempt"] == 1
    rendered = repr(repaired).lower()
    assert "ndcg" not in rendered
    assert "effect" not in rendered
    assert not is_mechanical_repair_failure(
        {"stage": "QUALIFY", "failure_class": "PROTOCOL", "reason_code": "X"}
    )
    assert not is_mechanical_repair_failure(
        {"stage": "API_CONTRACT", "failure_class": "INTERFACE", "effect": -0.1}
    )
    with pytest.raises(ValueError):
        build_mechanical_repair_request(
            original,
            {"stage": "RESOURCE", "failure_class": "RESOURCE", "reason_code": "OOM"},
        )


def test_candidate_local_multifile_package_materializes_with_actual_file_set(tmp_path: Path) -> None:
    spec = _spec()
    policy = SharedImplementerPolicy(
        allowed_files=CANDIDATE_LOCAL_ALLOWED_FILES,
        dependency_identity_ref="dependencies:fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref="runtime:fixture",
        runtime_identity_digest=_digest("runtime"),
        prompt_digest=_digest("prompt"),
        tool_policy_digest=_digest("tools"),
        implementation_token_ceiling=4096,
        execution_contract={"gpu_budget_gb": 10, "recbole_interface": "frozen"},
    )
    response = {
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "files": [
            {"path": "recclaw_ext/__init__.py", "content": "# candidate package\n"},
            {
                "path": "recclaw_ext/candidate.py",
                "content": (
                    "from recbole.model.general_recommender.bpr import BPR\n"
                    "from recclaw_ext.layers import helper\n\n"
                    "class FreshCandidateModel(BPR):\n"
                    "    pass\n"
                ),
            },
            {
                "path": "recclaw_ext/layers.py",
                "content": "def helper(value):\n    return value\n",
            },
        ],
        "implementation_summary": "candidate-local multi-file fixture",
    }
    schema = json.loads(
        (
            ROOT
            / "src/recclaw_core/experiments/helix_abc_v1/resources/q5_conversion_implementer_response_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(
        {
            "schema": "recclaw.research-line.q5-conversion-implementer-response.v1",
            "proposals": [response],
        },
        schema,
    )
    request = build_shared_implementer_request(spec, policy=policy)
    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=response,
        candidate_root=tmp_path / request["blind_candidate_id"],
        candidate_root_ref="conversion-fixture-root",
    )
    assert materialized.package.allowed_files == (
        "recclaw_ext/__init__.py",
        "recclaw_ext/candidate.py",
        "recclaw_ext/layers.py",
    )
    assert request["service_policy"]["execution_contract"]["gpu_budget_gb"] == 10


def test_conversion_schema_uses_provider_strict_subset_and_consumer_rejects_bad_files(
    tmp_path: Path,
) -> None:
    schema_path = (
        ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/resources/"
        "q5_conversion_implementer_response_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_provider_strict_schema(schema)

    spec = _spec()
    policy = SharedImplementerPolicy(
        allowed_files=CANDIDATE_LOCAL_ALLOWED_FILES,
        dependency_identity_ref="dependencies:fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref="runtime:fixture",
        runtime_identity_digest=_digest("runtime"),
        prompt_digest=_digest("prompt"),
        tool_policy_digest=_digest("tools"),
        implementation_token_ceiling=4096,
        execution_contract={"gpu_budget_gb": 10, "recbole_interface": "frozen"},
    )
    base_response = {
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "files": [
            {"path": "recclaw_ext/__init__.py", "content": "# package\n"},
            {"path": "recclaw_ext/candidate.py", "content": "class FreshCandidateModel: pass\n"},
        ],
        "implementation_summary": "fixture",
    }

    missing_init = {**base_response, "files": base_response["files"][1:]}
    with pytest.raises(InnovationSpineError, match="between two and five"):
        materialize_candidate_package(
            spec,
            policy=policy,
            implementation_response=missing_init,
            candidate_root=tmp_path / "missing-init",
            candidate_root_ref="conversion-fixture-root",
        )

    duplicate_path = {
        **base_response,
        "files": [*base_response["files"], base_response["files"][1]],
    }
    with pytest.raises(InnovationSpineError, match="duplicated"):
        materialize_candidate_package(
            spec,
            policy=policy,
            implementation_response=duplicate_path,
            candidate_root=tmp_path / "duplicate-path",
            candidate_root_ref="conversion-fixture-root",
        )


def test_conversion_screen_summary_keeps_seventeen_none_candidates_missing(tmp_path: Path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "q5a_idea_feasibility_runner",
        ROOT / "scripts/run_q5a_idea_feasibility.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    entries = []
    for index in range(17):
        realization_root = tmp_path / f"realization-{index:02d}"
        realization_root.mkdir()
        (realization_root / "MATCHED_EXECUTION_RECEIPT.json").write_text(
            json.dumps(
                {
                    "status": "NO_LEGAL_ADMITTED_ARM",
                    "baseline": None,
                    "candidate": None,
                    "stable": False,
                    "screen_signal": None,
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "candidate_id": f"candidate-{index}",
                "realization_root": realization_root,
            }
        )

    rows = runner._build_conversion_screen_results(entries, {}, set())

    assert len(rows) == 17
    assert all(row["status"] == "NO_LEGAL_ADMITTED_ARM" for row in rows)
    assert all(row["screen_cost_ms"] is None for row in rows)
    assert all(not row["stable"] for row in rows)


def test_fail_soft_batch_continues_after_one_arm_failure() -> None:
    def run_one(candidate_id: str) -> str:
        if candidate_id == "bad":
            raise RuntimeError("mechanical fixture failure")
        return "ok"

    rows = run_fail_soft_batch(("a", "bad", "c"), run_one)
    assert [row["candidate_id"] for row in rows] == ["a", "bad", "c"]
    assert rows[1]["status"] == "MISSING"
    assert rows[2]["status"] == "COMPLETED"


def test_screen_promotion_uses_fresh_full_seeds_and_shared_parent_counts() -> None:
    plan = build_conversion_execution_plan(
        ("a", "b", "c"), screen_seed=54303, full_seeds=(54304, 54305), promotion_limit=2
    )
    screens = (
        {"candidate_id": "a", "status": "COMPLETED_MATCHED_PAIR", "stable": True, "screen_signal": 0.2},
        {"candidate_id": "b", "status": "COMPLETED_MATCHED_PAIR", "stable": True, "screen_signal": 0.1},
        {"candidate_id": "c", "status": "MISSING", "stable": False, "screen_signal": 0.9},
    )
    assert choose_stable_promotions(screens, promotion_limit=2) == ("a", "b")
    finalized = finalize_conversion_execution_plan(plan, screens, ("a", "b"))
    assert finalized["run_counts"] == {
        "screen_candidate_runs": 3,
        "screen_parent_runs": 1,
        "full_candidate_runs": 4,
        "full_parent_runs": 2,
        "shared_parent_runs": 3,
        "screen_seed": 54303,
        "full_seeds": [54304, 54305],
    }


def test_stage_feasibility_head_preserves_17_arm_denominator_and_effect_shrinkage() -> None:
    labels = []
    for index in range(17):
        policies = ("STATIC",) if index < 8 else (("OUTCOME_AWARE",) if index < 13 else ("CURRENT_F1",))
        labels.append(
            {
                "policies": policies,
                "CONSTRUCT": "PASS",
                "MATERIALIZE": "PASS" if index != 16 else "MISSING",
                "QUALIFY": "PASS" if index < 10 else "MISSING",
                "RESOURCE_ADMITTED": "PASS" if index < 8 else "MISSING",
                "FULL_EPISODE": "PASS" if index < 2 else "MISSING",
            }
        )
    head = build_stage_feasibility_head(labels, policy_order=("STATIC", "CURRENT_F1", "OUTCOME_AWARE"))
    assert head["stage_denominator"] == 17
    assert head["policy_output"] == "NONUNIFORM"
    assert head["effect_head"] == {
        "observed_full_episode_count": 2,
        "shrinkage": "STRONG",
        "claim_allowed": False,
    }
