from __future__ import annotations

from dataclasses import replace

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    VersionedCapabilityRegistry,
)
from recclaw_core.experiments.helix_abc_v1.next_fresh_profile import (
    NextFreshProfileBuildError,
    NextFreshProfileBuildManifest,
    build_next_fresh_profile,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    CapabilityKindV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _capability(label: str) -> QualifiedCapabilityV1:
    return QualifiedCapabilityV1(
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref=f"semantics:{label}",
        semantic_identity_digest=_digest(f"{label}-semantics"),
        executable_entrypoint=f"recclaw_ext.{label}:{label.title()}Model",
        candidate_package_ref=f"candidate-package:{label}",
        candidate_package_digest=_digest(f"{label}-package"),
        source_tree_digest=_digest(f"{label}-tree"),
        qualification_receipt_ref=f"qualification-receipt:{label}",
        qualification_receipt_digest=_digest(f"{label}-qualification"),
        qualification_stage=QualificationStageV1.ONE_EPOCH_SMOKE,
        qualification_status=QualificationStatusV1.PASS,
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("protocol"),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
        predecessor_capability_ref=None,
        predecessor_capability_digest=None,
        current_campaign_ineligible=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


def _registry(
    capabilities: tuple[QualifiedCapabilityV1, ...],
) -> VersionedCapabilityRegistry:
    return VersionedCapabilityRegistry.build(
        registry_version="bc-wave-1",
        predecessor_registry_ref="registry:fixed-space-v2",
        predecessor_registry_digest=_digest("predecessor-registry"),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("protocol"),
        capabilities=capabilities,
    )


def _manifest(
    registry: VersionedCapabilityRegistry,
    *,
    reverse_entries: bool = False,
) -> NextFreshProfileBuildManifest:
    entries = (
        (
            "capability:existing-bpr",
            _digest("existing-bpr"),
            "recbole.model.general_recommender.bpr:BPR",
        ),
        (
            "capability:existing-lightgcn",
            _digest("existing-lightgcn"),
            "recbole.model.general_recommender.lightgcn:LightGCN",
        ),
    )
    if reverse_entries:
        entries = tuple(reversed(entries))
    return NextFreshProfileBuildManifest(
        profile_version="bc-wave-1-next",
        predecessor_profile_ref="profile:current-campaign",
        predecessor_profile_digest=_digest("current-profile"),
        current_campaign_profile_ref="profile:current-campaign",
        current_campaign_profile_digest=_digest("current-profile"),
        current_campaign_slate_ref="slate:current-campaign",
        current_campaign_slate_digest=_digest("current-slate"),
        predecessor_executable_entries=entries,
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        registry_version=registry.registry_version,
        protocol_ref=registry.protocol_ref,
        protocol_digest=registry.protocol_digest,
        compatibility_requirements=(
            "pairwise input",
            "general collaborative filtering",
        ),
    )


def test_deterministic_rebuild_is_order_independent_and_next_fresh_only() -> None:
    capability_a = _capability("candidate_a")
    capability_b = _capability("candidate_b")
    registry_ab = _registry((capability_a, capability_b))
    registry_ba = _registry((capability_b, capability_a))
    manifest_ab = _manifest(registry_ab)
    manifest_ba = _manifest(registry_ba, reverse_entries=True)
    current_campaign = {
        "profile_ref": manifest_ab.current_campaign_profile_ref,
        "profile_digest": manifest_ab.current_campaign_profile_digest,
        "slate_ref": manifest_ab.current_campaign_slate_ref,
        "slate_digest": manifest_ab.current_campaign_slate_digest,
    }
    current_before = canonical_json_bytes(current_campaign)

    profile_ab, receipt_ab = build_next_fresh_profile(
        manifest_ab,
        registry_ab,
    )
    profile_ba, receipt_ba = build_next_fresh_profile(
        manifest_ba,
        registry_ba,
    )
    rebuilt_profile, rebuilt_receipt = build_next_fresh_profile(
        manifest_ab,
        registry_ab,
    )

    assert manifest_ab.canonical_bytes() == manifest_ba.canonical_bytes()
    assert profile_ab.canonical_bytes() == profile_ba.canonical_bytes()
    assert receipt_ab.canonical_bytes() == receipt_ba.canonical_bytes()
    assert profile_ab.profile_id == profile_ba.profile_id
    assert profile_ab == rebuilt_profile
    assert receipt_ab == rebuilt_receipt
    assert receipt_ab.new_profile_ref == profile_ab.profile_id
    assert receipt_ab.new_profile_hash == profile_ab.digest
    assert receipt_ab.registry_ref == registry_ab.registry_id
    assert receipt_ab.registry_digest == registry_ab.digest
    assert receipt_ab.build_policy_ref == manifest_ab.manifest_id
    assert receipt_ab.build_policy_digest == manifest_ab.digest
    assert receipt_ab.current_profile_unchanged is True
    assert receipt_ab.deterministic_rebuild is True
    assert receipt_ab.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert profile_ab.current_campaign_eligible is False
    assert profile_ab.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert len(profile_ab.executable_entries) == 4
    assert canonical_json_bytes(current_campaign) == current_before


@pytest.mark.parametrize(
    ("field_name", "value", "reason"),
    (
        ("registry_ref", "registry:wrong", "identity drift"),
        ("registry_digest", _digest("wrong-registry"), "identity drift"),
        ("registry_version", "wrong-version", "version drift"),
        ("protocol_ref", "protocol:wrong", "protocol drift"),
        ("protocol_digest", _digest("wrong-protocol"), "protocol drift"),
    ),
)
def test_registry_identity_protocol_and_version_drift_fail_closed(
    field_name: str,
    value: str,
    reason: str,
) -> None:
    registry = _registry((_capability("candidate_a"),))
    manifest = replace(_manifest(registry), **{field_name: value})

    with pytest.raises(NextFreshProfileBuildError, match=reason):
        build_next_fresh_profile(manifest, registry)


def test_predecessor_identity_drift_fails_closed() -> None:
    registry = _registry((_capability("candidate_a"),))

    with pytest.raises(
        NextFreshProfileBuildError,
        match="predecessor identity",
    ):
        replace(
            _manifest(registry),
            predecessor_profile_digest=_digest("wrong-predecessor"),
        )


def test_conflicting_predecessor_entry_fails_closed() -> None:
    registry = _registry((_capability("candidate_a"),))
    manifest = _manifest(registry)
    first = manifest.predecessor_executable_entries[0]
    conflict = (first[0], _digest("conflict"), first[2])

    with pytest.raises(
        NextFreshProfileBuildError,
        match="conflicting predecessor",
    ):
        replace(
            manifest,
            predecessor_executable_entries=(first, conflict),
        )
