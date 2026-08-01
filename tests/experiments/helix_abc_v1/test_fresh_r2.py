from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    AVAILABLE_DEPENDENCIES,
    BUDGET_LIMITS,
    PROTOCOL_REQUIREMENTS,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (
    R2_PROPOSAL_SEEDS,
    _r2_bindings,
    _r2_environment,
    build_active_r2_profile,
    build_r1_registry,
    derive_fresh_r2_proposal_schema,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
    render_r2_proposal_prompt,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    project_open_producer_draft,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchProfileEntryOriginV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityResolutionResultV1,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"


def _draft(*, semantics_digest: str | None) -> dict[str, object]:
    expressible = semantics_digest is not None
    return {
        "producer_role": "mechanism_composer",
        "hypothesis": "A routed multi-channel interaction can improve ranking.",
        "mechanism_change": "Use the selected trainable interaction mechanism.",
        "competing_explanation": "Any change is caused only by extra capacity.",
        "matched_control_requirement": "Compare with BPR under the same seed.",
        "implementation_requirements": [
            "RecBole general recommender interface",
            "candidate-local package",
        ],
        "expected_evidence": ["A real matched development run closes."],
        "falsifier": "Reject the mechanism account if the comparison does not close.",
        "compatibility_requirements": list(PROTOCOL_REQUIREMENTS),
        "high_change_justification": (
            "An active mechanism is reused without a novelty claim."
            if expressible
            else "A new trainable interaction is required."
        ),
        "current_profile_expressibility_claim": (
            "EXPRESSIBLE" if expressible else "NOT_EXPRESSIBLE"
        ),
        "resolution_facts": {
            "requested_current_semantics_digest": semantics_digest,
            "capability_diff": [] if expressible else ["new interaction relation"],
            "high_change_dimensions": [] if expressible else ["INTERACTION_STRUCTURE"],
            "required_dependencies": list(AVAILABLE_DEPENDENCIES),
            "required_budget": dict(BUDGET_LIMITS),
        },
    }


def test_real_r1_registry_activates_all_qualified_capabilities_origin_blind() -> None:
    artifacts, _receipt = load_registered_r1_artifacts(ROOT)
    registry = build_r1_registry(artifacts)
    current, _manifest, _next_profile, _build_receipt, active = (
        build_active_r2_profile(registry)
    )
    catalog = public_active_profile_catalog(
        active, artifacts, seed=R2_PROPOSAL_SEEDS[0]
    )
    prompt = render_r2_proposal_prompt(
        (RESOURCE_ROOT / "fresh_r2_producer_prompt_v1.txt").read_text(
            encoding="utf-8"
        ),
        slot_id="slot-01",
        seed=R2_PROPOSAL_SEEDS[0],
        role="mechanism_composer",
        active=active,
        catalog=catalog,
    )

    assert len(artifacts) == 11
    assert sum(item.episode_payload is not None for item in artifacts) == 7
    assert len(current.entries) == 66
    assert len(active.entries) == 77
    assert sum(
        entry.origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        for entry in active.entries
    ) == 11
    assert "side_a" not in prompt
    assert "side_b" not in prompt
    assert "R1_FRESH" not in prompt


def test_r2_schema_and_existing_resolver_preserve_search_ready_and_innovation() -> None:
    artifacts, _receipt = load_registered_r1_artifacts(ROOT)
    registry = build_r1_registry(artifacts)
    _current, _manifest, _next_profile, _build_receipt, active = (
        build_active_r2_profile(registry)
    )
    schema = derive_fresh_r2_proposal_schema()
    validate_provider_strict_schema(schema)
    selected = artifacts[0].capability
    expressible_response = {
        "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
        "proposals": [_draft(semantics_digest=selected.semantic_identity_digest)],
    }
    innovation_response = {
        "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
        "proposals": [_draft(semantics_digest=None)],
    }
    jsonschema.validate(expressible_response, schema)
    jsonschema.validate(innovation_response, schema)

    expressible_spec, expressible_facts = project_open_producer_draft(
        expressible_response["proposals"][0], bindings=_r2_bindings(active)
    )
    expressible_resolution = resolve_capability(
        expressible_spec,
        resolution_facts=expressible_facts,
        environment=_r2_environment(active),
    )
    innovation_spec, innovation_facts = project_open_producer_draft(
        innovation_response["proposals"][0], bindings=_r2_bindings(active)
    )
    innovation_resolution = resolve_capability(
        innovation_spec,
        resolution_facts=innovation_facts,
        environment=_r2_environment(active),
    )

    assert expressible_resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
    assert expressible_resolution.resolved_current_capability_ref == selected.capability_id
    assert innovation_resolution.resolution is (
        CapabilityResolutionResultV1.INNOVATION_REQUIRED
    )
