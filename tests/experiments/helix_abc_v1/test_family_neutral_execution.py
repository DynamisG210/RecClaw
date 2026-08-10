from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from campaign_train_worker import _validate_recipe_identity
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanism,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    ExperimentBindingV1,
    render_campaign_worker_command,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    RecBoleQualificationFixture,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    SharedImplementerPolicy,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    QualificationStageV1,
    QualificationStatusV1,
    RealizationModeV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_orchestration import (
    qualify_local_innovation_candidate,
)
from recclaw_core.experiments.helix_abc_v1 import fresh_r1


PROFILE_REF = "test:campaign-profile"
PROFILE_DIGEST = sha256_digest({"profile": PROFILE_REF})
DATASET_MANIFEST_DIGEST = "1" * 64
RUNTIME_BINDING_DIGEST = "2" * 64
RUNTIME_RELEASE_DIGEST = "3" * 64
FILESYSTEM_CAPABILITY_DIGEST = "4" * 64
PERMIT_DIGEST = "5" * 64

FAMILY_MODULES = {
    "BPR": "bpr",
    "LightGCN": "lightgcn",
    "LINE": "line",
}
FAMILY_DATA_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "innovation_spine" / "data"
)
FAMILY_RUNTIME_REF = "runtime:family-neutral-qualification"
FAMILY_RUNTIME_DIGEST = sha256_digest({"runtime": FAMILY_RUNTIME_REF})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _family_spec(base_model_config: str) -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis=(
            f"The {base_model_config} execution substrate can host a fresh "
            "candidate-local GeneralRecommender."
        ),
        mechanism_change="Exercise the declared family through one shared qualifier.",
        competing_explanation="The qualifier may accidentally instantiate a fixed family.",
        matched_control_requirement="Use the declared existing base model configuration.",
        implementation_requirements=("Candidate-local GeneralRecommender entrypoint.",),
        expected_evidence=("Development-only staged qualification receipt.",),
        falsifier="Reject non-pairwise or non-finite GeneralRecommender behavior.",
        compatibility_requirements=("RecBole general recommender", "PAIRWISE input"),
        protocol_ref="protocol:family-neutral-qualification",
        protocol_digest=sha256_digest({"protocol": "family-neutral"}),
        context_ref="context:family-neutral-qualification",
        context_digest=sha256_digest({"context": "family-neutral"}),
        current_profile_ref="profile:family-neutral-qualification",
        current_profile_digest=sha256_digest({"profile": "family-neutral"}),
        producer_role="mechanism_composer",
        high_change_justification="This is a qualification-only family fixture.",
        current_profile_expressibility_claim=CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE,
        idea_mode=IdeaModeV1.FRONTIER_HYPOTHESIS,
        research_question="Does the selected family satisfy the shared behavioral contract?",
        closest_parent=base_model_config,
        minimal_testable_wedge="Construct, backpropagate, score, and full-sort one mini batch.",
        causal_chain=("declared base config", "candidate construction", "shared API checks"),
        discriminative_predictions=("construction uses the declared base config",),
        mechanism_off_definition="Any implementation that fails the shared API contract.",
        resource_hypothesis="The mini fixture is sufficient for one local smoke epoch.",
        realization_mode=RealizationModeV1.PARENT_PRESERVING,
        execution_contract={
            "capability_family": f"TEST_{base_model_config}_FAMILY",
            "model": "FreshCandidateModel",
            "base_model_config": base_model_config,
            "config": {},
        },
    )


def _family_policy(spec: OpenResearchSpecV1) -> SharedImplementerPolicy:
    base_model_config = str(spec.execution_contract["base_model_config"])
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/__init__.py", "recclaw_ext/candidate.py"),
        dependency_identity_ref=f"dependencies:{base_model_config}",
        dependency_identity_digest=sha256_digest(
            {"base_model_config": base_model_config, "dataset": "mini"}
        ),
        runtime_identity_ref=FAMILY_RUNTIME_REF,
        runtime_identity_digest=FAMILY_RUNTIME_DIGEST,
        prompt_digest=sha256_digest({"prompt": "family-neutral"}),
        tool_policy_digest=sha256_digest({"tool_policy": "candidate-local"}),
        implementation_token_ceiling=4096,
        execution_contract=spec.execution_contract,
    )


def _family_response(base_model_config: str) -> dict[str, Any]:
    module_name = FAMILY_MODULES[base_model_config]
    source = (
        f"from recbole.model.general_recommender.{module_name} import "
        f"{base_model_config}\n\n"
        f"class FreshCandidateModel({base_model_config}):\n"
        "    pass\n"
    )
    return {
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "files": [
            {"content": "# candidate-local package\n", "path": "recclaw_ext/__init__.py"},
            {"content": source, "path": "recclaw_ext/candidate.py"},
        ],
        "implementation_summary": (
            f"Family-neutral {base_model_config} GeneralRecommender fixture."
        ),
    }


def _family_fixture(tmp_path: Path) -> RecBoleQualificationFixture:
    return RecBoleQualificationFixture(
        project_root=Path(__file__).resolve().parents[3],
        recbole_root=_recbole_root(),
        data_path=FAMILY_DATA_ROOT,
        dataset="mini",
        # Deliberately mirror the old controller default. The qualifier must
        # use the OpenResearchSpec execution_contract instead.
        base_model_config="BPR",
        seed=20260805,
        checkpoint_dir=tmp_path / "qualification" / "checkpoints",
        runtime_identity_ref=FAMILY_RUNTIME_REF,
        runtime_identity_digest=FAMILY_RUNTIME_DIGEST,
    )


def _catalog_recipe(
    mechanism_id: str,
    *,
    execution_role: str = "CANDIDATE",
) -> dict[str, object]:
    mechanism = executable_mechanism(mechanism_id)
    recipe = mechanism.execution_recipe()
    recipe.update(
        {
            "capability_family": mechanism.base_mechanism_id,
            "capability_ref": mechanism.mechanism_id,
            "capability_digest": mechanism.execution_recipe_digest,
            "dataset": COMMON_DATASET,
            "evaluator": dict(COMMON_EVALUATOR),
            "execution_role": execution_role,
            "profile_digest": PROFILE_DIGEST,
            "profile_ref": PROFILE_REF,
            "split": COMMON_SPLIT,
        }
    )
    if execution_role == "COMPARATOR":
        comparator_ref = f"test:comparator:{mechanism.mechanism_id}"
        recipe.update(
            {
                "comparator_digest": sha256_digest(
                    {
                        "entrypoint": mechanism.entrypoint,
                        "model": mechanism.model,
                    }
                ),
                "comparator_ref": comparator_ref,
            }
        )
    return recipe


def _binding(
    recipe: dict[str, object],
    *,
    run_id: str,
    candidate_root: Path | None = None,
) -> ExperimentBindingV1:
    return ExperimentBindingV1.from_execution_recipe(
        recipe,
        candidate_root=candidate_root,
        dataset_manifest_digest=DATASET_MANIFEST_DIGEST,
        seed=2026,
        epochs=100,
        timeout_seconds=1500,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        resource_telemetry=False,
        watchdog_seconds=None,
        prefix_contract_digest=None,
        run_id=run_id,
        round_id=f"test-round:{run_id}",
        claim_id=f"test-claim:{run_id}",
        permit_digest=PERMIT_DIGEST,
        runtime_binding_digest=RUNTIME_BINDING_DIGEST,
        runtime_release_digest=RUNTIME_RELEASE_DIGEST,
        runner_abi="test.runner.v1",
        filesystem_capability_digest=FILESYSTEM_CAPABILITY_DIGEST,
    )


def _command(binding: ExperimentBindingV1, root: Path) -> list[str]:
    return render_campaign_worker_command(
        binding,
        python_executable=root / "python",
        worker_path=root / "campaign_train_worker.py",
        checkpoint_dir=root / "checkpoints",
        data_path=root / "data",
        execution_recipe_path=root / "execution_recipe.json",
        filesystem_capability_path=root / "filesystem_capability.json",
        log_path=root / "training.log",
        output_path=root / "worker_result.json",
        project_root=root / "project",
        recbole_root=root / "RecBole",
        start_confirmation_path=root / "start_confirmation.json",
        start_gate_path=root / "start_gate.json",
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_two_legal_families_use_one_command_and_common_full_sort_contract(
    tmp_path: Path,
) -> None:
    bpr = _binding(_catalog_recipe("BPR_MF"), run_id="bpr")
    lightgcn = _binding(_catalog_recipe("LIGHTGCN"), run_id="lightgcn")
    common_root = tmp_path / "common"
    bpr_command = _command(bpr, common_root)
    lightgcn_command = _command(lightgcn, common_root)

    assert bpr_command[1] == lightgcn_command[1]
    assert _option(bpr_command, "--model") == bpr.model
    assert _option(lightgcn_command, "--model") == lightgcn.model
    assert _option(bpr_command, "--model") != _option(lightgcn_command, "--model")
    assert _option(lightgcn_command, "--model") != "BPR"
    assert _option(bpr_command, "--dataset") == COMMON_DATASET
    assert _option(lightgcn_command, "--dataset") == COMMON_DATASET
    assert bpr.evaluator == lightgcn.evaluator == COMMON_EVALUATOR
    assert bpr.evaluator["candidate_universe"] == "FULL_SORT"
    assert bpr.evaluator["metric"] == "NDCG@10"
    assert _option(bpr_command, "--binding-digest") == bpr.digest
    assert _option(lightgcn_command, "--binding-digest") == lightgcn.digest
    assert bpr.worker_recipe()["model"] == bpr.model
    assert lightgcn.worker_recipe()["model"] == lightgcn.model


def test_explicit_bpr_comparator_is_typed_as_comparator() -> None:
    binding = _binding(
        _catalog_recipe("BPR_MF", execution_role="COMPARATOR"),
        run_id="matched-bpr",
    )

    assert binding.execution_role == "COMPARATOR"
    assert binding.comparator_ref == "test:comparator:BPR_MF"
    assert binding.comparator_digest is not None
    assert binding.capability_family == "BPR_MF"


def test_missing_recipe_fails_before_filesystem_mutation_or_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_root = tmp_path / "side"

    def fail_if_launched(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("worker launch was attempted")

    monkeypatch.setattr(fresh_r1.subprocess, "Popen", fail_if_launched)
    with pytest.raises(
        fresh_r1.FreshR1Error,
        match="explicit execution_recipe is required",
    ):
        fresh_r1.run_development_training(
            repo_root=tmp_path,
            side_root=side_root,
            run_id="missing-recipe",
            seed=2026,
            candidate_root=None,
            entrypoint="recbole.model.general_recommender.bpr:BPR",
            source_sha256="a" * 64,
            execution_recipe=None,
        )
    assert not side_root.exists()


def test_binding_semantic_digest_is_distinct_from_newline_physical_hash() -> None:
    binding = _binding(_catalog_recipe("LIGHTGCN"), run_id="semantic-persistence")

    semantic_digest = binding.digest
    physical_digest = bytes_sha256(
        canonical_json_bytes(binding.canonical_dict()) + b"\n"
    )

    assert physical_digest != semantic_digest
    assert binding.digest == semantic_digest


def test_complete_binding_round_trips_from_runner_payload() -> None:
    binding = _binding(_catalog_recipe("LIGHTGCN"), run_id="round-trip")

    restored = ExperimentBindingV1.from_canonical_dict(binding.canonical_dict())

    assert restored == binding
    assert restored.digest == binding.digest


def test_worker_rejects_binding_digest_mismatch_before_model_import() -> None:
    recipe = {
        "entrypoint": "recbole.model.general_recommender.bpr:BPR",
        "execution_binding_digest": "a" * 64,
        "mechanism_id": "BPR_MF",
        "model": "BPR",
    }

    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_recipe_identity(
            recipe,
            model="BPR",
            binding_digest="b" * 64,
        )

    _validate_recipe_identity(
        recipe,
        model="BPR",
        binding_digest="a" * 64,
    )


@pytest.mark.parametrize(
    "base_model_config",
    ("BPR", "LightGCN", "LINE"),
)
def test_staged_innovation_qualification_is_general_and_spec_selected(
    tmp_path: Path,
    base_model_config: str,
) -> None:
    spec = _family_spec(base_model_config)
    policy = _family_policy(spec)
    candidate_parent = tmp_path / "candidate-parent"
    candidate_parent.mkdir()

    from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
        build_shared_implementer_request,
    )

    request = build_shared_implementer_request(spec, policy=policy)
    candidate_root = candidate_parent / str(request["blind_candidate_id"])

    def bpr_only_comparator(*_args: object) -> None:
        raise AssertionError("BPR/LightGCN comparator evidence is optional")

    _materialized, qualification = qualify_local_innovation_candidate(
        spec,
        policy=policy,
        implementation_response=_family_response(base_model_config),
        candidate_root=candidate_root,
        candidate_root_ref=f"candidate-root:{base_model_config}",
        fixture=_family_fixture(tmp_path),
        unit_check=bpr_only_comparator,
    )

    assert qualification.receipt.status is QualificationStatusV1.PASS
    assert qualification.receipt.stage is QualificationStageV1.ONE_EPOCH_SMOKE
    construction = qualification.stage_observations["CONSTRUCTION"]
    assert construction["config_model"] == base_model_config
    assert construction["model_class"] == "FreshCandidateModel"

    unit = qualification.stage_observations["UNIT"]
    assert unit["model_type"] == "general"
    assert unit["input_type"] == "pairwise"
    assert unit["calculate_loss_scalars"] >= 1
    assert unit["backward_parameter_count"] > 0
    assert unit["predict_values"] > 0
    assert unit["full_sort_values"] > unit["predict_values"]
    assert unit["optional_unit_check"]["status"] == "FAIL_OPTIONAL"
    assert unit["shared_unit_check"] == "FAIL_OPTIONAL"
