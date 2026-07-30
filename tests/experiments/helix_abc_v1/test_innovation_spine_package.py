from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    content_id,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    candidate_tree_identity,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    InnovationSpineError,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
    origin_blind_projection,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
)


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _spec(
    *,
    producer_role: str,
    context_ref: str,
) -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis=(
            "A gated residual graph propagation mechanism should preserve "
            "short-range preference while reducing oversmoothing."
        ),
        mechanism_change=(
            "Add a learned residual gate around each graph propagation layer."
        ),
        competing_explanation=(
            "Any improvement may come from additional parameters rather than "
            "the propagation gate."
        ),
        matched_control_requirement=(
            "Match the parent embedding size, depth, optimizer, and parameter count."
        ),
        implementation_requirements=(
            "Implement a candidate-local RecBole GeneralRecommender.",
            "Use the standard pairwise trainer contract.",
        ),
        expected_evidence=(
            "Construction and API qualification receipts.",
            "One-epoch development smoke completion.",
        ),
        falsifier=(
            "Reject the mechanism if the gated model cannot satisfy the frozen "
            "pairwise RecBole interface."
        ),
        compatibility_requirements=(
            "Frozen general-CF development protocol.",
            "RecBole 1.2.1 standard trainer.",
        ),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("protocol"),
        context_ref=context_ref,
        context_digest=_digest(context_ref),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=_digest("current-profile"),
        producer_role=producer_role,
        high_change_justification=(
            "The learned layer-wise gate is not an exact one- or two-operator "
            "composition in the frozen 66-program catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


@pytest.fixture
def policy() -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/gated_fixture.py",),
        dependency_identity_ref="dependencies:recbole-bpr-fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref="runtime:recclaw-frozen-recbole",
        runtime_identity_digest=_digest("runtime"),
        prompt_digest=_digest("shared-prompt"),
        tool_policy_digest=_digest("write-only-tool-policy"),
        implementation_token_ceiling=4096,
    )


def _implementation_response(
    *,
    path: str = "recclaw_ext/gated_fixture.py",
) -> dict[str, Any]:
    return {
        "entrypoint": "recclaw_ext.gated_fixture:GatedFixtureModel",
        "files": [
            {
                "content": (
                    "from recbole.model.general_recommender.bpr import BPR\n\n"
                    "class GatedFixtureModel(BPR):\n"
                    "    pass\n"
                ),
                "path": path,
            }
        ],
        "implementation_summary": (
            "Deterministic local fixture for the shared implementation boundary."
        ),
    }


def _request_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_request_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_request_keys(item))
    return keys


def test_equivalent_a_b_specs_share_one_blind_request(
    policy: SharedImplementerPolicy,
) -> None:
    spec_a = _spec(
        producer_role="mechanism_composer",
        context_ref="context:controller-a",
    )
    spec_b = _spec(
        producer_role="frontier_architect",
        context_ref="context:controller-b",
    )

    projection_a = origin_blind_projection(spec_a)
    projection_b = origin_blind_projection(spec_b)
    request_a = build_shared_implementer_request(spec_a, policy=policy)
    request_b = build_shared_implementer_request(spec_b, policy=policy)

    assert spec_a.digest != spec_b.digest
    assert projection_a == projection_b
    assert request_a == request_b
    assert request_a["blind_candidate_id"] == request_b["blind_candidate_id"]
    forbidden = {
        "arm",
        "context_digest",
        "context_ref",
        "controller",
        "metric_values",
        "origin",
        "outcome",
        "producer_role",
        "research_spec_digest",
        "research_spec_ref",
        "result",
        "spec_id",
    }
    assert _request_keys(request_a).isdisjoint(forbidden)
    rendered = canonical_json_bytes(request_a).decode("utf-8")
    assert spec_a.producer_role not in rendered
    assert spec_b.producer_role not in rendered
    assert spec_a.context_ref not in rendered
    assert spec_b.context_ref not in rendered
    assert spec_a.spec_id not in rendered
    assert spec_b.spec_id not in rendered
    assert '"outcome"' not in rendered
    assert '"metric_values"' not in rendered


def test_fresh_exclusive_materialization_closes_package_identity(
    policy: SharedImplementerPolicy,
    tmp_path: Path,
) -> None:
    spec = _spec(
        producer_role="mechanism_composer",
        context_ref="context:controller-a",
    )
    request = build_shared_implementer_request(spec, policy=policy)
    candidate_root = tmp_path / str(request["blind_candidate_id"])
    candidate_root_ref = "candidate-root:" + str(request["blind_candidate_id"])

    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=_implementation_response(),
        candidate_root=candidate_root,
        candidate_root_ref=candidate_root_ref,
    )

    package = materialized.package
    manifest = snapshot_candidate_tree(candidate_root)
    source_digest, root_digest = candidate_tree_identity(
        candidate_root,
        candidate_root_ref=candidate_root_ref,
    )
    assert candidate_root.is_dir()
    assert not candidate_root.is_symlink()
    assert tuple(row["path"] for row in manifest) == policy.allowed_files
    assert all(not (candidate_root / row["path"]).is_symlink() for row in manifest)
    assert package.allowed_files == policy.allowed_files
    assert package.source_tree_digest == source_digest
    assert package.candidate_root_digest == root_digest
    assert package.research_spec_ref == spec.spec_id
    assert package.research_spec_digest == spec.digest
    assert package.protocol_ref == spec.protocol_ref
    assert package.protocol_digest == spec.protocol_digest
    assert package.origin_blind_projection_digest == sha256_digest(
        materialized.blind_projection
    )
    assert package.implementation_receipt_digest == sha256_digest(
        materialized.implementation_receipt
    )
    assert package.implementation_receipt_ref == content_id(
        "recclaw-implementation-receipt-v1",
        materialized.implementation_receipt,
    )
    assert materialized.implementation_receipt["request_digest"] == (
        sha256_digest(materialized.shared_request)
    )
    assert materialized.implementation_receipt["source_tree_digest"] == (
        package.source_tree_digest
    )
    assert replace(package).package_id == package.package_id
    persisted = json.loads(
        canonical_json_bytes(package.canonical_dict()).decode("utf-8")
    )
    assert persisted == package.canonical_dict()

    before = tuple(
        path.read_bytes()
        for path in candidate_root.rglob("*")
        if path.is_file()
    )
    with pytest.raises(InnovationSpineError) as captured:
        materialize_candidate_package(
            spec,
            policy=policy,
            implementation_response=_implementation_response(),
            candidate_root=candidate_root,
            candidate_root_ref=candidate_root_ref,
        )
    assert captured.value.failure_class == "PACKAGE"
    assert captured.value.reason_code == "CANDIDATE_ROOT_NOT_FRESH"
    after = tuple(
        path.read_bytes()
        for path in candidate_root.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_allowlist_failure_is_implementation_class_and_creates_no_root(
    policy: SharedImplementerPolicy,
    tmp_path: Path,
) -> None:
    spec = _spec(
        producer_role="mechanism_composer",
        context_ref="context:controller-a",
    )
    request = build_shared_implementer_request(spec, policy=policy)
    candidate_root = tmp_path / str(request["blind_candidate_id"])
    response = _implementation_response(path="recclaw_ext/forbidden.py")

    with pytest.raises(InnovationSpineError) as captured:
        materialize_candidate_package(
            spec,
            policy=policy,
            implementation_response=response,
            candidate_root=candidate_root,
            candidate_root_ref="candidate-root:forbidden",
        )
    assert captured.value.failure_class == "IMPLEMENTATION"
    assert captured.value.reason_code == "IMPLEMENTATION_FILE_INVALID"
    assert not candidate_root.exists()


def test_symlink_root_is_package_class_and_target_is_untouched(
    policy: SharedImplementerPolicy,
    tmp_path: Path,
) -> None:
    spec = _spec(
        producer_role="mechanism_composer",
        context_ref="context:controller-a",
    )
    request = build_shared_implementer_request(spec, policy=policy)
    target = tmp_path / "existing-target"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    candidate_root = tmp_path / str(request["blind_candidate_id"])
    candidate_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(InnovationSpineError) as captured:
        materialize_candidate_package(
            spec,
            policy=policy,
            implementation_response=_implementation_response(),
            candidate_root=candidate_root,
            candidate_root_ref="candidate-root:symlink",
        )
    assert captured.value.failure_class == "PACKAGE"
    assert captured.value.reason_code == "CANDIDATE_ROOT_SYMLINK"
    assert marker.read_text(encoding="utf-8") == "preserve"
