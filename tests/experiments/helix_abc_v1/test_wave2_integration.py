from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import (
    D1_ACCEPTED_INTAKE,
    E0_ACCEPTED_INTAKE,
    F0_ACCEPTED_INTAKE,
    WAVE1_ACCEPTED_COMMIT,
    WAVE1_ACCEPTED_TREE,
    FrozenR1R2PrefreezeManifestV1,
    Wave2IntegrationError,
    Wave2IntegrationHarnessV1,
    Wave2OwnerCorrectionRequired,
    Wave2OwnerIntakeV1,
    Wave2OwnerLaneV1,
    accepted_wave2_harness,
    dry_run_r1_r2_launcher,
    load_prefreeze_manifest,
    prefreeze_missing_fields,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)


ROOT = Path(__file__).resolve().parents[3]
PREFREEZE_TEMPLATE = (
    ROOT / "docs" / "research_line" / "vnext" / "R1_R2_PREFREEZE_TEMPLATE.json"
)
RC1_CANDIDATE = (
    ROOT / "docs" / "research_line" / "vnext" / "WAVE1_RC1_CANDIDATE.json"
)
WAVE2_CHECKLIST = (
    ROOT / "docs" / "research_line" / "vnext" / "WAVE2_INTAKE_CHECKLIST.json"
)
INTAKE_RECEIPTS = (
    ROOT / "docs" / "research_line" / "vnext" / "D1_F0_INTAKE_RECEIPTS.json"
)
E0_INTAKE_RECEIPT = (
    ROOT / "docs" / "research_line" / "vnext" / "E0_INTAKE_RECEIPT.json"
)
WAVE2_MODULE = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "wave2_integration.py"
)


def _digest(label: str) -> str:
    return sha256_digest({"wave2_fixture": label})


def _intake(lane: Wave2OwnerLaneV1, *, index: int) -> Wave2OwnerIntakeV1:
    return Wave2OwnerIntakeV1(
        lane=lane,
        accepted_commit=f"{index + 1:040x}",
        parent_commit=f"{index + 11:040x}",
        owner_file_manifest_sha256=_digest(f"{lane.value}-bytes"),
        targeted_tests_receipt_sha256=_digest(f"{lane.value}-tests"),
        structure_lint_receipt_sha256=_digest(f"{lane.value}-lint"),
        public_entrypoint=f"owner_fixture.lane_{index}:public_adapter",
    )


def _template() -> dict[str, Any]:
    return json.loads(PREFREEZE_TEMPLATE.read_bytes())


def _complete_manifest() -> dict[str, Any]:
    payload = _template()

    def fill(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            return {
                key: fill(item, f"{path}.{key}" if path else key)
                for key, item in value.items()
            }
        if value is not None:
            return value
        if path == "source_identity.commit":
            return "a" * 40
        if path == "shared_proposal_call.token_budget":
            return 4096
        if path == "shared_proposal_call.call_count":
            return 1
        if path == "shared_proposal_call.granularity":
            return "PER_PROPOSAL"
        if path.endswith("_digest"):
            return _digest(path)
        if path.endswith("_ref"):
            return f"fixture:{path}"
        raise AssertionError(f"unhandled fixture path: {path}")

    return fill(payload)


def _write_canonical(path: Path, payload: dict[str, Any]) -> str:
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    return bytes_sha256(raw)


def test_wave1_candidate_manifest_binds_locally_accepted_identity() -> None:
    payload = json.loads(RC1_CANDIDATE.read_bytes())

    assert payload["accepted_commit"] == WAVE1_ACCEPTED_COMMIT
    assert payload["accepted_tree"] == WAVE1_ACCEPTED_TREE
    assert payload["status"] == "LOCAL_WAVE1_ACCEPTED_RC1_CANDIDATE"
    assert payload["wave1_gate"] == {
        "canonical_integrated_test": "2 passed",
        "independent_gate": "PASS",
        "required_boundaries_closed": 7,
    }
    assert payload["wave2_owner_intakes"]["e0_search_adapter"] == {
        "accepted_commit": E0_ACCEPTED_INTAKE.accepted_commit,
        "g_merge_commit": "e6ac840a76fa8a50880df935bc93c0fb473b262d",
    }
    assert payload["wave2_owner_intakes"]["d1_scientific_episode_adapter"] == {
        "accepted_commit": D1_ACCEPTED_INTAKE.accepted_commit,
        "g_merge_commit": "3fe57ba8a1c50334907eb2878d70b8664e402617",
    }
    assert payload["wave2_owner_intakes"]["f0_open_meta_interface"] == {
        "accepted_commit": F0_ACCEPTED_INTAKE.accepted_commit,
        "g_merge_commit": "5489f73d196b29d1c6602d7f7ea14f9f417774e8",
    }
    assert payload["external_actions"] == {
        "pushed": False,
        "released": False,
        "tagged": False,
    }


def test_wave2_intake_checklist_records_all_accepted_owner_lanes() -> None:
    payload = json.loads(WAVE2_CHECKLIST.read_bytes())

    assert payload["accepted_base_commit"] == WAVE1_ACCEPTED_COMMIT
    assert set(payload["lanes"]) == {lane.value for lane in Wave2OwnerLaneV1}
    assert payload["lanes"]["E0_SEARCH_ADAPTER"]["accepted_intake"] == {
        **E0_ACCEPTED_INTAKE.canonical_dict(),
        "fixture_sha256": (
            "1bebe835a9e079d33f0e5025912788ca2694128a1fb74b7278ccd0e532ced445"
        ),
        "g_merge_commit": "e6ac840a76fa8a50880df935bc93c0fb473b262d",
    }
    assert (
        payload["lanes"]["D1_SCIENTIFIC_EPISODE_ADAPTER"]["accepted_intake"]
        == {
            **D1_ACCEPTED_INTAKE.canonical_dict(),
            "g_merge_commit": "3fe57ba8a1c50334907eb2878d70b8664e402617",
        }
    )
    assert (
        payload["lanes"]["F0_OPEN_META_INTERFACE"]["accepted_intake"]
        == {
            **F0_ACCEPTED_INTAKE.canonical_dict(),
            "g_merge_commit": "5489f73d196b29d1c6602d7f7ea14f9f417774e8",
        }
    )
    assert payload["required_accepted_intake_fields"] == [
        "accepted_commit",
        "parent_commit",
        "owner_file_manifest_sha256",
        "targeted_tests_receipt_sha256",
        "structure_lint_receipt_sha256",
        "public_entrypoint",
    ]
    assert payload["merge_policy"] == (
        "preserve_owner_commit_and_parent_identity"
    )


def test_d1_f0_intake_receipt_digests_bind_recorded_evidence() -> None:
    payload = json.loads(INTAKE_RECEIPTS.read_bytes())
    expected = {
        "D1_SCIENTIFIC_EPISODE_ADAPTER": D1_ACCEPTED_INTAKE,
        "F0_OPEN_META_INTERFACE": F0_ACCEPTED_INTAKE,
    }

    for lane, intake in expected.items():
        receipt = payload[lane]
        assert receipt["accepted_commit"] == intake.accepted_commit
        assert receipt["parent_commit"] == intake.parent_commit
        assert receipt["public_entrypoint"] == intake.public_entrypoint
        assert (
            sha256_digest(receipt["owner_file_manifest"])
            == intake.owner_file_manifest_sha256
        )
        assert (
            sha256_digest(receipt["targeted_tests_receipt"])
            == intake.targeted_tests_receipt_sha256
        )
        assert (
            sha256_digest(receipt["structure_lint_receipt"])
            == intake.structure_lint_receipt_sha256
        )


def test_e0_intake_receipt_digests_bind_recorded_evidence() -> None:
    receipt = json.loads(E0_INTAKE_RECEIPT.read_bytes())

    assert receipt["accepted_commit"] == E0_ACCEPTED_INTAKE.accepted_commit
    assert receipt["parent_commit"] == E0_ACCEPTED_INTAKE.parent_commit
    assert receipt["public_entrypoint"] == E0_ACCEPTED_INTAKE.public_entrypoint
    assert (
        sha256_digest(receipt["owner_file_manifest"])
        == E0_ACCEPTED_INTAKE.owner_file_manifest_sha256
    )
    assert (
        sha256_digest(receipt["targeted_tests_receipt"])
        == E0_ACCEPTED_INTAKE.targeted_tests_receipt_sha256
    )
    assert (
        sha256_digest(receipt["structure_lint_receipt"])
        == E0_ACCEPTED_INTAKE.structure_lint_receipt_sha256
    )


def test_accepted_wave2_harness_is_complete() -> None:
    harness = accepted_wave2_harness()

    assert harness.ready is True
    assert harness.missing_lanes == ()
    assert {intake.lane for intake in harness.intakes} == {
        Wave2OwnerLaneV1.E0_SEARCH_ADAPTER,
        Wave2OwnerLaneV1.D1_SCIENTIFIC_EPISODE_ADAPTER,
        Wave2OwnerLaneV1.F0_OPEN_META_INTERFACE,
    }


def test_wave2_module_has_no_owner_runtime_or_external_effect_imports() -> None:
    tree = ast.parse(WAVE2_MODULE.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_modules <= {
        "__future__",
        "dataclasses",
        "enum",
        "json",
        "pathlib",
        "re",
        "typing",
        "canonical",
    }
    assert not {
        "open_spec",
        "innovation_spine",
        "innovation_recbole_adapter",
        "next_fresh_profile",
        "scientific_episode",
        "state_store",
        "subprocess",
        "socket",
        "requests",
        "torch",
        "recbole",
    } & imported_modules


def test_wave2_harness_attaches_only_exact_owner_entrypoints() -> None:
    harness = Wave2IntegrationHarnessV1()
    intakes = [
        _intake(lane, index=index)
        for index, lane in enumerate(Wave2OwnerLaneV1)
    ]

    assert harness.missing_lanes == tuple(Wave2OwnerLaneV1)
    assert harness.ready is False
    for intake in intakes:
        harness = harness.attach(
            intake,
            observed_entrypoint=intake.public_entrypoint,
        )

    assert harness.ready is True
    assert harness.missing_lanes == ()
    assert {
        item["lane"] for item in harness.canonical_dict()["intakes"]
    } == {lane.value for lane in Wave2OwnerLaneV1}


def test_wave2_harness_returns_minimal_owner_correction() -> None:
    intake = _intake(Wave2OwnerLaneV1.E0_SEARCH_ADAPTER, index=0)

    with pytest.raises(Wave2OwnerCorrectionRequired) as exc_info:
        Wave2IntegrationHarnessV1().attach(
            intake,
            observed_entrypoint="owner_fixture.search:wrong_adapter",
        )

    error = exc_info.value
    assert error.lane is Wave2OwnerLaneV1.E0_SEARCH_ADAPTER
    assert error.field == "public_entrypoint"
    assert error.expected == intake.public_entrypoint
    assert error.observed == "owner_fixture.search:wrong_adapter"


def test_checked_in_prefreeze_template_keeps_unknown_identities_unset() -> None:
    payload = _template()
    missing = prefreeze_missing_fields(payload)

    assert "provider_identity.endpoint_ref" in missing
    assert "provider_identity.model_digest" in missing
    assert "r1_identity.seed_ref" in missing
    assert "r2_identity.outcome_namespace_digest" in missing
    assert payload["shared_proposal_call"]["proposal_budget_per_side"] == 8
    assert payload["shared_proposal_call"]["no_retry"] is True
    assert payload["evidence_policy"]["held_out_absent"] is True
    assert payload["r1_gate"]["result_slots"] == {
        "fresh_spec_receipt_refs": [],
        "producer_role_receipt_refs": [],
        "qualified_capability_receipt_refs": [],
        "real_mechanism_change_receipt_refs": [],
    }


def test_complete_prefreeze_dry_run_is_deterministic_and_provider_free(
    tmp_path: Path,
) -> None:
    payload = _complete_manifest()
    path = tmp_path / "prefreeze.json"
    digest = _write_canonical(path, payload)

    loaded = load_prefreeze_manifest(path, expected_digest=digest)
    first = dry_run_r1_r2_launcher(path, expected_digest=digest)
    second = dry_run_r1_r2_launcher(path, expected_digest=digest)

    assert isinstance(loaded, FrozenR1R2PrefreezeManifestV1)
    assert loaded.digest == digest
    assert first == second
    assert first["provider_calls"] == 0
    assert first["experiment_runs"] == 0
    assert first["outcomes_consumed"] == 0
    assert first["launch_authorized"] is False
    assert first["r1_r2_isolation_verified"] is True
    assert (
        first["r1_gate_result_slots"]
        == "UNPOPULATED_REAL_R1_RECEIPTS_ONLY"
    )


def test_prefreeze_fails_closed_on_missing_required_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete.json"
    digest = _write_canonical(path, _template())

    with pytest.raises(
        Wave2IntegrationError,
        match="unresolved required fields",
    ):
        load_prefreeze_manifest(path, expected_digest=digest)


def test_prefreeze_fails_closed_on_byte_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "prefreeze.json"
    _write_canonical(path, _complete_manifest())

    with pytest.raises(Wave2IntegrationError, match="byte digest mismatch"):
        load_prefreeze_manifest(path, expected_digest="f" * 64)


def test_prefreeze_fails_closed_on_noncanonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "prefreeze.json"
    raw = json.dumps(_complete_manifest(), indent=2).encode("utf-8")
    path.write_bytes(raw)

    with pytest.raises(Wave2IntegrationError, match="not canonical"):
        load_prefreeze_manifest(path, expected_digest=bytes_sha256(raw))


def test_prefreeze_fails_closed_on_r1_r2_identity_collision() -> None:
    payload = _complete_manifest()
    payload["r2_identity"]["seed_ref"] = payload["r1_identity"]["seed_ref"]

    with pytest.raises(Wave2IntegrationError, match="must be isolated"):
        FrozenR1R2PrefreezeManifestV1(payload=payload)


@pytest.mark.parametrize(
    ("path", "invalid_value", "message"),
    [
        ("shared_proposal_call.no_retry", False, "must remain frozen"),
        ("shared_proposal_call.proposal_budget_per_side", 7, "must remain frozen"),
        ("evidence_policy.held_out_absent", False, "must remain frozen"),
        (
            "shared_implementation.manual_candidate_patch_forbidden",
            False,
            "must remain frozen",
        ),
        (
            "shared_implementation.qualification_evidence_class",
            "SCIENTIFIC",
            "must remain frozen",
        ),
    ],
)
def test_prefreeze_fails_closed_when_fixed_policy_is_weakened(
    path: str,
    invalid_value: Any,
    message: str,
) -> None:
    payload = _complete_manifest()
    target: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = invalid_value

    with pytest.raises(Wave2IntegrationError, match=message):
        FrozenR1R2PrefreezeManifestV1(payload=payload)


def test_prefreeze_rejects_any_pre_outcome_gate_result() -> None:
    payload = copy.deepcopy(_complete_manifest())
    payload["r1_gate"]["result_slots"]["fresh_spec_receipt_refs"] = [
        "smoke:not-real-r1"
    ]

    with pytest.raises(Wave2IntegrationError, match="only real R1 receipts"):
        FrozenR1R2PrefreezeManifestV1(payload=payload)
