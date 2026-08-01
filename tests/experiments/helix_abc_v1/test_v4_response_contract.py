from __future__ import annotations

import json
import ast
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from recclaw_core.experiments.helix_abc_v1.v4_response_contract import (
    V1_UNIQUE_SCHEMA_PATHS,
    V4_LOCAL_ARRAY_PATHS,
    V4LocalUniquenessError,
    derive_v4_provider_schema,
    validate_v4_local_uniqueness,
    validate_v4_response_contract,
    validate_v4_schema_derivation,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v2 import (
    V4_ATTEMPT_RECEIPT_REL,
    V4_AUTH_REL,
    V4_BLOCKED_REL,
    V4_MANIFEST_REL,
    V4_NEGATIVE_FIXTURE_REL,
    V4_POLICY_REL,
    V4_PROVIDER_SCHEMA_REL,
    V4_READY_REL,
    V4_VALIDATOR_REL,
    V4_VALID_FIXTURE_REL,
    expected_v4_authorization,
    expected_v4_negative_fixture,
    expected_v4_provider_schema,
    expected_v4_retry_policy,
    expected_v4_valid_fixture,
    provider_free_v4_dry_run,
    validate_prefreeze_v4,
    validate_prefreeze_v4_payload,
    verify_v3_seal,
)
from recclaw_core.experiments.helix_abc_v1.wave2_integration import (
    Wave2IntegrationError,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = (
    ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
V1_SCHEMA_PATH = (
    RESOURCE_ROOT / "fresh_open_spec_proposal_response_v1.schema.json"
)
SENTINEL_PATH = RESOURCE_ROOT / "fresh_open_spec_schema_probe_payload_v1.json"


def _v1_schema() -> dict[str, object]:
    return json.loads(V1_SCHEMA_PATH.read_bytes())


def _sentinel() -> dict[str, object]:
    return json.loads(SENTINEL_PATH.read_bytes())


def _value_at(proposal: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = proposal
    for part in path:
        assert isinstance(current, dict)
        current = current[part]
    return current


def _set_at(
    proposal: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current: dict[str, object] = proposal
    for part in path[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = value


def test_v4_schema_derivation_removes_exactly_six_unique_items_keywords() -> None:
    v1 = _v1_schema()
    v4 = derive_v4_provider_schema(v1)

    assert len(V1_UNIQUE_SCHEMA_PATHS) == 6
    validate_v4_schema_derivation(v1_schema=v1, v4_provider_schema=v4)
    restored = deepcopy(v4)
    for path in V1_UNIQUE_SCHEMA_PATHS:
        current = restored
        for part in path[:-1]:
            current = current[part]
        current[path[-1]] = True
    assert restored == v1


def test_v4_valid_sentinel_passes_provider_and_local_contract() -> None:
    sentinel = _sentinel()
    provider_schema = derive_v4_provider_schema(_v1_schema())

    validate_v4_response_contract(sentinel, provider_schema=provider_schema)


@pytest.mark.parametrize("path", V4_LOCAL_ARRAY_PATHS)
def test_v4_rejects_every_original_unique_array_path(path: tuple[str, ...]) -> None:
    response = _sentinel()
    proposal = response["proposals"][0]
    assert isinstance(proposal, dict)
    values = _value_at(proposal, path)
    assert isinstance(values, list)
    duplicate = values[0] if values else "qualification-only duplicate sentinel"
    _set_at(proposal, path, [duplicate, deepcopy(duplicate)])
    provider_schema = derive_v4_provider_schema(_v1_schema())

    jsonschema.validate(response, provider_schema)
    with pytest.raises(V4LocalUniquenessError) as caught:
        validate_v4_response_contract(response, provider_schema=provider_schema)
    assert caught.value.path == path
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(response, _v1_schema())


@pytest.mark.parametrize("path", V4_LOCAL_ARRAY_PATHS)
def test_v4_local_validator_rejects_missing_or_non_array(
    path: tuple[str, ...],
) -> None:
    response = _sentinel()
    proposal = response["proposals"][0]
    assert isinstance(proposal, dict)
    _set_at(proposal, path, "not-an-array")
    with pytest.raises(V4LocalUniquenessError) as non_array:
        validate_v4_local_uniqueness(response)
    assert non_array.value.reason == "NOT_ARRAY"

    response = _sentinel()
    proposal = response["proposals"][0]
    assert isinstance(proposal, dict)
    current = proposal
    for part in path[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    del current[path[-1]]
    with pytest.raises(V4LocalUniquenessError) as missing:
        validate_v4_local_uniqueness(response)
    assert missing.value.reason == "MISSING_FIELD"


def test_v4_local_uniqueness_uses_canonical_deep_value_equality() -> None:
    response = _sentinel()
    proposal = response["proposals"][0]
    assert isinstance(proposal, dict)
    left = {"nested": [1, {"a": True, "b": None}]}
    right = {"nested": [1, {"b": None, "a": True}]}
    _set_at(
        proposal,
        ("compatibility_requirements",),
        [left, right],
    )

    with pytest.raises(V4LocalUniquenessError) as caught:
        validate_v4_local_uniqueness(response)
    assert caught.value.reason == "DUPLICATE_CANONICAL_VALUE"


def test_v3_and_v1_schema_are_sealed_before_v4() -> None:
    sealed = verify_v3_seal(ROOT)
    assert len(sealed) == 6
    for relative, digest in sealed.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_checked_in_v4_schema_and_fixtures_are_exact_derivations() -> None:
    provider_path = ROOT / V4_PROVIDER_SCHEMA_REL
    valid_path = ROOT / V4_VALID_FIXTURE_REL
    negative_path = ROOT / V4_NEGATIVE_FIXTURE_REL

    provider = json.loads(provider_path.read_bytes())
    valid = json.loads(valid_path.read_bytes())
    negative = json.loads(negative_path.read_bytes())
    assert provider_path.read_bytes() == canonical_json_bytes(provider)
    assert valid_path.read_bytes() == canonical_json_bytes(valid)
    assert negative_path.read_bytes() == canonical_json_bytes(negative)
    assert provider == expected_v4_provider_schema(ROOT)
    assert valid == expected_v4_valid_fixture(ROOT)
    assert negative == expected_v4_negative_fixture(ROOT)
    jsonschema.validate(negative, provider)
    with pytest.raises(V4LocalUniquenessError):
        validate_v4_response_contract(negative, provider_schema=provider)


def test_v4_manifest_policy_authorization_and_dry_run_are_fail_closed() -> None:
    policy = json.loads((ROOT / V4_POLICY_REL).read_bytes())
    authorization = json.loads((ROOT / V4_AUTH_REL).read_bytes())
    manifest = validate_prefreeze_v4(ROOT)
    dry_run = provider_free_v4_dry_run(ROOT)

    assert policy == expected_v4_retry_policy()
    assert authorization == expected_v4_authorization(ROOT)
    equivalence = manifest["response_contract_equivalence"]
    assert equivalence["v1_schema_digest"] == bytes_sha256(
        V1_SCHEMA_PATH.read_bytes()
    )
    assert len(equivalence["removed_schema_paths"]) == 6
    assert equivalence["semantic_equivalence"] == "EXACT_V1_ACCEPTANCE_SET"
    assert equivalence["local_validator_code_digest"] == bytes_sha256(
        (ROOT / V4_VALIDATOR_REL).read_bytes()
    )
    future = manifest["future_r1_scientific_contract"]
    assert future["proposal_slots_per_side"] == 8
    assert future["proposal_denominator_per_side"] == 8
    assert future["side_a_call_contract_digest"] == future[
        "shared_call_contract_digest"
    ]
    assert future["side_b_call_contract_digest"] == future[
        "shared_call_contract_digest"
    ]
    assert dry_run["provider_calls"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["held_out_reads"] == 0


def test_v4_validator_rejects_semantic_equivalence_mutation() -> None:
    payload = json.loads((ROOT / V4_MANIFEST_REL).read_bytes())
    payload["response_contract_equivalence"]["semantic_equivalence"] = "RELAXED"
    with pytest.raises(Wave2IntegrationError):
        validate_prefreeze_v4_payload(ROOT, payload)


def test_v4_probe_orders_local_contract_before_any_downstream_use() -> None:
    path = ROOT / "scripts/reprobe_fresh_open_spec_endpoint_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    local_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "validate_v4_response_contract"
    ]
    provider_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call_with_session"
    ]
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert len(local_calls) == 2
    assert len(provider_calls) == 4
    assert not any(
        name.endswith(
            (
                "open_spec",
                "innovation_spine",
                "innovation_recbole_adapter",
                "capability_admission",
            )
        )
        for name in imports
    )


def test_checked_in_v4_outcome_is_bounded_and_side_effect_free() -> None:
    attempt_path = ROOT / V4_ATTEMPT_RECEIPT_REL
    if not attempt_path.exists():
        pytest.skip("V4 schema qualification has not been executed")
    receipt = json.loads(attempt_path.read_bytes())
    attempts = receipt["physical_attempts"]

    assert 1 <= len(attempts) <= 3
    assert receipt["physical_provider_calls"] == len(attempts)
    assert receipt["retry_count"] == len(attempts) - 1
    assert len({item["request_envelope_digest"] for item in attempts}) == 1
    assert len({item["request_payload_digest"] for item in attempts}) == 1
    for field in (
        "candidate_admissions",
        "candidate_qualifications",
        "candidate_roots_created",
        "held_out_reads",
        "open_specs_projected",
        "outcomes_consumed",
        "research_candidates_generated",
        "resolver_calls",
        "training_runs",
    ):
        assert receipt[field] == 0
    if (ROOT / V4_BLOCKED_REL).exists():
        blocked = json.loads((ROOT / V4_BLOCKED_REL).read_bytes())
        assert receipt["final_classification"] == (
            "HTTP_200_RESPONSE_CONTRACT_VALUE_ERROR"
        )
        assert receipt["physical_provider_calls"] == 1
        assert receipt["retry_count"] == 0
        assert attempts[0]["local_uniqueness_status"] == "NOT_REACHED"
        assert blocked["final_classification"] == receipt[
            "final_classification"
        ]
        assert blocked["r1_worker_launch_authorized"] is False
    assert not ((ROOT / V4_BLOCKED_REL).exists() and (ROOT / V4_READY_REL).exists())
