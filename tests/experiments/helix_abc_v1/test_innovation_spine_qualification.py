from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import (
    innovation_recbole_adapter as adapter_module,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterV1,
    RecBoleQualificationFixture,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    MaterializedCandidate,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (
    runtime_release_digest,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DATA_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "innovation_spine"
    / "data"
)
RUNTIME_REF = "runtime:recclaw-frozen-recbole"


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _research_spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis=(
            "A candidate-local pairwise recommender should satisfy the common "
            "RecBole qualification contract."
        ),
        mechanism_change="Introduce a candidate-local model implementation.",
        competing_explanation="A pass may only reflect the inherited BPR substrate.",
        matched_control_requirement="Use the frozen BPR development configuration.",
        implementation_requirements=(
            "Provide one GeneralRecommender entrypoint.",
            "Use the standard pairwise interaction contract.",
        ),
        expected_evidence=(
            "Identity-closed qualification receipt.",
            "One-epoch development smoke completion.",
        ),
        falsifier="Reject any candidate that fails the shared RecBole qualifier.",
        compatibility_requirements=(
            "Frozen local general-CF protocol.",
            "RecBole 1.2.1 runtime.",
        ),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("integration-protocol"),
        context_ref="context:local-integration-fixture",
        context_digest=_digest("integration-context"),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=_digest("current-profile"),
        producer_role="mechanism_composer",
        high_change_justification=(
            "The fixture represents a candidate outside the frozen catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _policy() -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/integrated_fixture.py",),
        dependency_identity_ref="dependencies:recbole-bpr-fixture",
        dependency_identity_digest=_digest("integration-dependencies"),
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
        prompt_digest=_digest("integration-prompt"),
        tool_policy_digest=_digest("integration-tool-policy"),
        implementation_token_ceiling=4096,
    )


def _response(*, api_invalid: bool) -> dict[str, Any]:
    imports = "from recbole.model.general_recommender.bpr import BPR\n"
    body = ""
    class_name = "IntegratedFixtureModel"
    if api_invalid:
        imports += "from recbole.utils import InputType\n"
        body = "    input_type = InputType.POINTWISE\n"
        class_name = "InvalidIntegratedFixtureModel"
    return {
        "entrypoint": f"recclaw_ext.integrated_fixture:{class_name}",
        "files": [
            {
                "content": (
                    imports
                    + "\n"
                    + f"class {class_name}(BPR):\n"
                    + (body or "    pass\n")
                ),
                "path": "recclaw_ext/integrated_fixture.py",
            }
        ],
        "implementation_summary": "Fresh local integration fixture.",
    }


def _materialize(
    tmp_path: Path,
    *,
    api_invalid: bool,
    label: str,
) -> tuple[
    OpenResearchSpecV1,
    MaterializedCandidate,
    Path,
    SharedImplementerPolicy,
]:
    spec = _research_spec()
    policy = _policy()
    request = build_shared_implementer_request(spec, policy=policy)
    parent = tmp_path / label
    parent.mkdir()
    candidate_root = parent / str(request["blind_candidate_id"])
    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=_response(api_invalid=api_invalid),
        candidate_root=candidate_root,
        candidate_root_ref=(
            "candidate-root:" + label + ":" + str(request["blind_candidate_id"])
        ),
    )
    return spec, materialized, candidate_root, policy


def _qualification_fixture(
    tmp_path: Path,
    *,
    label: str,
) -> RecBoleQualificationFixture:
    return RecBoleQualificationFixture(
        project_root=ROOT,
        recbole_root=_recbole_root(),
        data_path=FIXTURE_DATA_ROOT,
        dataset="mini",
        base_model_config="BPR",
        seed=20260730,
        checkpoint_dir=tmp_path / "qualification" / label / "checkpoints",
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
    )


def _unit_check(model: object, config: object, dataset: object) -> None:
    from recbole.utils import ModelType

    assert config["MODEL_TYPE"] is ModelType.GENERAL
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert dataset.item_num > 1


def _assert_development_only(result: Any) -> None:
    assert result.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert result.receipt.mechanism_belief_authority == "NONE"
    assert result.receipt.current_campaign_effect_evidence is False


def test_materialized_package_runs_through_one_epoch_with_closed_receipt(
    tmp_path: Path,
) -> None:
    spec, materialized, candidate_root, _policy_value = _materialize(
        tmp_path,
        api_invalid=False,
        label="valid",
    )
    package = materialized.package
    before = snapshot_candidate_tree(candidate_root)

    result = MechanicalRecBoleAdapterV1().qualify(
        package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=_qualification_fixture(tmp_path, label="valid"),
        unit_check=_unit_check,
    )

    receipt = result.receipt
    assert receipt.status is QualificationStatusV1.PASS
    assert receipt.stage is QualificationStageV1.ONE_EPOCH_SMOKE
    assert receipt.candidate_package_ref == package.package_id
    assert receipt.candidate_package_digest == package.digest
    assert receipt.research_spec_ref == spec.spec_id
    assert receipt.research_spec_digest == spec.digest
    assert receipt.candidate_root_ref == package.candidate_root_ref
    assert receipt.candidate_root_digest == package.candidate_root_digest
    assert receipt.source_tree_digest == package.source_tree_digest
    assert receipt.protocol_ref == spec.protocol_ref
    assert receipt.protocol_digest == spec.protocol_digest
    assert receipt.runtime_identity_ref == package.runtime_identity_ref
    assert receipt.runtime_identity_digest == package.runtime_identity_digest
    assert (
        result.stage_observations["STATIC_VALIDATION"]["entrypoint_source"]
        == "recclaw_ext/integrated_fixture.py"
    )
    assert (
        result.stage_observations["STATIC_VALIDATION"]["entrypoint_class"]
        == "IntegratedFixtureModel"
    )
    assert result.smoke_executions == 1
    assert snapshot_candidate_tree(candidate_root) == before
    _assert_development_only(result)


def test_api_invalid_package_stops_at_first_interface_failure(
    tmp_path: Path,
) -> None:
    spec, materialized, candidate_root, _policy_value = _materialize(
        tmp_path,
        api_invalid=True,
        label="api-invalid",
    )
    before = snapshot_candidate_tree(candidate_root)

    result = MechanicalRecBoleAdapterV1().qualify(
        materialized.package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=_qualification_fixture(tmp_path, label="api-invalid"),
        unit_check=_unit_check,
    )

    receipt = result.receipt
    assert receipt.status is QualificationStatusV1.FAIL
    assert receipt.stage is QualificationStageV1.API_CONTRACT
    assert receipt.failure_class is QualificationFailureClassV1.INTERFACE
    assert receipt.static_result is QualificationCheckStatusV1.PASS
    assert receipt.construction_result is QualificationCheckStatusV1.PASS
    assert receipt.api_contract_result is QualificationCheckStatusV1.FAIL
    assert receipt.unit_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.smoke_result is QualificationCheckStatusV1.NOT_RUN
    assert result.failure_detail["reason_code"] == "INPUT_TYPE_MISMATCH"
    assert result.smoke_executions == 0
    assert snapshot_candidate_tree(candidate_root) == before
    _assert_development_only(result)


@pytest.mark.parametrize(
    ("mutation", "failure_class", "reason_code"),
    (
        (
            {"research_spec_digest": _digest("wrong-spec")},
            QualificationFailureClassV1.IMPLEMENTATION,
            "PACKAGE_SPEC_BINDING_MISMATCH",
        ),
        (
            {"protocol_digest": _digest("wrong-protocol")},
            QualificationFailureClassV1.PROTOCOL,
            "PACKAGE_PROTOCOL_BINDING_MISMATCH",
        ),
        (
            {"runtime_identity_digest": _digest("wrong-runtime")},
            QualificationFailureClassV1.RUNTIME,
            "PACKAGE_RUNTIME_BINDING_MISMATCH",
        ),
        (
            {"executable_entrypoint": "recclaw_ext.other:OtherModel"},
            QualificationFailureClassV1.IMPLEMENTATION,
            "ENTRYPOINT_OUTSIDE_PACKAGE",
        ),
    ),
)
def test_package_binding_drift_is_rejected_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, str],
    failure_class: QualificationFailureClassV1,
    reason_code: str,
) -> None:
    spec, materialized, candidate_root, _policy_value = _materialize(
        tmp_path,
        api_invalid=False,
        label="binding-drift",
    )
    package = replace(materialized.package, **mutation)
    before = snapshot_candidate_tree(candidate_root)
    construction_calls = 0

    def forbidden_construction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal construction_calls
        construction_calls += 1
        raise AssertionError("construction must not run for package drift")

    monkeypatch.setattr(
        adapter_module,
        "_construct_runtime",
        forbidden_construction,
    )
    result = MechanicalRecBoleAdapterV1().qualify(
        package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=_qualification_fixture(tmp_path, label="binding-drift"),
        unit_check=_unit_check,
    )

    receipt = result.receipt
    assert receipt.status is QualificationStatusV1.FAIL
    assert receipt.stage is QualificationStageV1.STATIC_VALIDATION
    assert receipt.failure_class is failure_class
    assert receipt.static_result is QualificationCheckStatusV1.FAIL
    assert receipt.construction_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.api_contract_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.unit_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.smoke_result is QualificationCheckStatusV1.NOT_RUN
    assert result.failure_detail["reason_code"] == reason_code
    assert construction_calls == 0
    assert result.smoke_executions == 0
    assert snapshot_candidate_tree(candidate_root) == before
    _assert_development_only(result)


def test_tree_identity_drift_is_rejected_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, materialized, candidate_root, _policy_value = _materialize(
        tmp_path,
        api_invalid=False,
        label="tree-drift",
    )
    source_path = candidate_root / "recclaw_ext" / "integrated_fixture.py"
    source_path.write_bytes(source_path.read_bytes() + b"\n# drift\n")
    drifted = snapshot_candidate_tree(candidate_root)
    construction_calls = 0

    def forbidden_construction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal construction_calls
        construction_calls += 1
        raise AssertionError("construction must not run for tree drift")

    monkeypatch.setattr(
        adapter_module,
        "_construct_runtime",
        forbidden_construction,
    )
    result = MechanicalRecBoleAdapterV1().qualify(
        materialized.package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=_qualification_fixture(tmp_path, label="tree-drift"),
        unit_check=_unit_check,
    )

    receipt = result.receipt
    assert receipt.status is QualificationStatusV1.FAIL
    assert receipt.stage is QualificationStageV1.STATIC_VALIDATION
    assert receipt.failure_class is QualificationFailureClassV1.IMPLEMENTATION
    assert receipt.static_result is QualificationCheckStatusV1.FAIL
    assert receipt.construction_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.api_contract_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.unit_result is QualificationCheckStatusV1.NOT_RUN
    assert receipt.smoke_result is QualificationCheckStatusV1.NOT_RUN
    assert (
        result.failure_detail["reason_code"]
        == "CANDIDATE_TREE_DIGEST_MISMATCH"
    )
    assert construction_calls == 0
    assert result.smoke_executions == 0
    assert snapshot_candidate_tree(candidate_root) == drifted
    _assert_development_only(result)
