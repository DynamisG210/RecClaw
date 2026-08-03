from __future__ import annotations

import json
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import EVIDENCE_CLASS
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterV1,
    RecBoleQualificationFixture,
    candidate_tree_identity,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (
    runtime_release_digest,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NO_MECHANISM_BELIEF_AUTHORITY,
    CandidatePackageV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "innovation_spine"
)


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _research_spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="The local fixture should satisfy the frozen RecBole contract.",
        mechanism_change="Use a candidate-local pairwise recommender entrypoint.",
        competing_explanation="A pass may only reflect inherited BPR behavior.",
        matched_control_requirement="Use the exact frozen BPR configuration.",
        implementation_requirements=("Provide a GeneralRecommender entrypoint.",),
        expected_evidence=("Produce development-only qualification receipts.",),
        falsifier="Reject any fixture that fails the shared interface.",
        compatibility_requirements=("RecBole 1.2.1 standard trainer.",),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=sha256_digest({"fixture": "protocol"}),
        context_ref="context:local-qualification-fixture",
        context_digest=sha256_digest({"fixture": "context"}),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=sha256_digest({"fixture": "profile"}),
        producer_role="mechanism_composer",
        high_change_justification=(
            "The fixture stands in for a candidate outside the frozen catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _package(
    candidate_root: Path,
    *,
    class_name: str,
    research_spec: OpenResearchSpecV1,
    candidate_module: str = "recclaw_ext.fixture_model",
    candidate_file: str = "recclaw_ext/fixture_model.py",
) -> CandidatePackageV1:
    candidate_root_ref = "fixture-root:" + candidate_root.name
    source_digest, root_digest = candidate_tree_identity(
        candidate_root,
        candidate_root_ref=candidate_root_ref,
    )
    return CandidatePackageV1(
        research_spec_ref=research_spec.spec_id,
        research_spec_digest=research_spec.digest,
        protocol_ref=research_spec.protocol_ref,
        protocol_digest=research_spec.protocol_digest,
        source_tree_digest=source_digest,
        candidate_root_ref=candidate_root_ref,
        candidate_root_digest=root_digest,
        executable_entrypoint=(
            f"{candidate_module}:{class_name}"
        ),
        allowed_files=(candidate_file,),
        dependency_identity_ref="dependencies:recbole-bpr-fixture",
        dependency_identity_digest=sha256_digest(
            {"base_model_config": "BPR", "dataset": "mini"}
        ),
        runtime_identity_ref="runtime:recclaw-frozen-recbole",
        runtime_identity_digest=runtime_release_digest(),
        implementation_receipt_ref="implementation:fixture",
        implementation_receipt_digest=sha256_digest(
            {"fixture": "implementation"}
        ),
        origin_blind_projection_digest=sha256_digest(
            {"fixture": "origin-blind"}
        ),
    )


@pytest.fixture
def qualification_fixture(tmp_path: Path) -> RecBoleQualificationFixture:
    return RecBoleQualificationFixture(
        project_root=ROOT,
        recbole_root=_recbole_root(),
        data_path=FIXTURE_ROOT / "data",
        dataset="mini",
        base_model_config="BPR",
        seed=20260730,
        checkpoint_dir=tmp_path / "checkpoints",
        runtime_identity_ref="runtime:recclaw-frozen-recbole",
        runtime_identity_digest=runtime_release_digest(),
    )


def _shared_unit_check(model: object, config: object, dataset: object) -> None:
    import torch
    from recbole.utils import ModelType

    parameters = tuple(model.parameters())
    assert parameters
    assert sum(parameter.numel() for parameter in parameters) > 0
    assert all(isinstance(parameter, torch.Tensor) for parameter in parameters)
    assert config["MODEL_TYPE"] is ModelType.GENERAL
    assert dataset.item_num > 1


def test_real_recbole_vertical_slice_emits_development_only_receipt(
    qualification_fixture: RecBoleQualificationFixture,
    tmp_path: Path,
) -> None:
    candidate_root = FIXTURE_ROOT / "valid_candidate"
    research_spec = _research_spec()
    package = _package(
        candidate_root,
        class_name="QualifiedFixtureModel",
        research_spec=research_spec,
    )

    result = MechanicalRecBoleAdapterV1().qualify(
        package,
        research_spec=research_spec,
        candidate_root=candidate_root,
        fixture=qualification_fixture,
        unit_check=_shared_unit_check,
    )

    receipt = result.receipt
    assert (
        receipt.stage is QualificationStageV1.ONE_EPOCH_SMOKE
    ), result.to_dict()
    assert receipt.status is QualificationStatusV1.PASS
    assert receipt.failure_class is QualificationFailureClassV1.NONE
    assert receipt.static_result is QualificationCheckStatusV1.PASS
    assert receipt.construction_result is QualificationCheckStatusV1.PASS
    assert receipt.api_contract_result is QualificationCheckStatusV1.PASS
    assert receipt.unit_result is QualificationCheckStatusV1.PASS
    assert receipt.smoke_result is QualificationCheckStatusV1.PASS
    assert receipt.evidence_class == EVIDENCE_CLASS == "DEVELOPMENT_ONLY"
    assert (
        receipt.mechanism_belief_authority
        == NO_MECHANISM_BELIEF_AUTHORITY
        == "NONE"
    )
    assert receipt.current_campaign_effect_evidence is False
    assert result.smoke_executions == 1
    assert result.failure_detail is None
    construction = result.stage_observations["CONSTRUCTION"]
    assert construction["config_model"] == "BPR"
    assert construction["dataset_class"] == "Dataset"
    assert construction["model_class"] == "QualifiedFixtureModel"
    assert construction["model_type"] == "general"
    api = result.stage_observations["API_CONTRACT"]
    assert api["input_type"] == "pairwise"
    assert api["model_type"] == "general"
    assert api["input_fields"] == ["item_id", "neg_item_id", "user_id"]
    assert api["calculate_loss_scalars"] == 1
    assert api["predict_values"] > 0
    assert api["full_sort_values"] > api["predict_values"]
    assert (
        result.stage_observations["ONE_EPOCH_SMOKE"][
            "metric_values_excluded_from_qualification"
        ]
        is True
    )

    receipt_path = tmp_path / "qualification_receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes() + b"\n")
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == receipt.canonical_dict()
    rendered = canonical_json_bytes(result.to_dict()).decode("utf-8")
    assert "mechanism_interpretation" not in rendered
    assert "best_valid" not in rendered


def test_invalid_input_fixture_stops_before_unit_and_smoke(
    qualification_fixture: RecBoleQualificationFixture,
) -> None:
    candidate_root = FIXTURE_ROOT / "invalid_candidate"
    research_spec = _research_spec()
    package = _package(
        candidate_root,
        class_name="InvalidInputFixtureModel",
        research_spec=research_spec,
    )

    result = MechanicalRecBoleAdapterV1().qualify(
        package,
        research_spec=research_spec,
        candidate_root=candidate_root,
        fixture=qualification_fixture,
        unit_check=_shared_unit_check,
    )

    receipt = result.receipt
    assert receipt.stage is QualificationStageV1.API_CONTRACT
    assert receipt.status is QualificationStatusV1.FAIL
    assert receipt.failure_class is QualificationFailureClassV1.INTERFACE
    assert receipt.static_result is QualificationCheckStatusV1.PASS
    assert receipt.construction_result is QualificationCheckStatusV1.PASS
    assert receipt.api_contract_result is QualificationCheckStatusV1.FAIL
    assert receipt.unit_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.smoke_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert receipt.mechanism_belief_authority == "NONE"
    assert receipt.current_campaign_effect_evidence is False
    assert result.failure_detail == {
        "error_type": "_QualificationStageFailure",
        "failure_class": "INTERFACE",
        "reason_code": "INPUT_TYPE_MISMATCH",
        "stage": "API_CONTRACT",
    }
    assert result.smoke_executions == 0
    assert "ONE_EPOCH_SMOKE" not in result.stage_observations


def test_candidate_local_missing_declared_time_field_stays_typed_failure(
    qualification_fixture: RecBoleQualificationFixture,
) -> None:
    candidate_root = FIXTURE_ROOT / "missing_time_field_candidate"
    research_spec = _research_spec()
    package = _package(
        candidate_root,
        class_name="MissingDeclaredTimeFieldModel",
        candidate_module="recclaw_ext.fixture_model",
        candidate_file="recclaw_ext/fixture_model.py",
        research_spec=research_spec,
    )

    result = MechanicalRecBoleAdapterV1().qualify(
        package,
        research_spec=research_spec,
        candidate_root=candidate_root,
        fixture=qualification_fixture,
        unit_check=_shared_unit_check,
    )

    assert result.receipt.stage is QualificationStageV1.CONSTRUCTION
    assert result.receipt.failure_class is QualificationFailureClassV1.INTERFACE
    assert result.failure_detail == {
        "error_type": "_QualificationStageFailure",
        "failure_class": "INTERFACE",
        "reason_code": "KEYERROR",
        "stage": "CONSTRUCTION",
    }
    assert result.smoke_executions == 0
