from __future__ import annotations

from types import SimpleNamespace

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
    bind_search_candidate,
)
from recclaw_core.research_line.portfolio import (
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)
from recclaw_core.research_line.runtime import (
    RoundCandidateHandoffV1,
    _route_metadata_for_attempt,
)
from recclaw_core.research_line.interfaces import ProducerOutcome

from test_runtime import _context, _fixed_proposals
from test_task_memory_credit_v2 import _spec


def test_route_metadata_carries_handoff_compute_and_resource_identity() -> None:
    profile = adapt_current_search_profile(
        campaign_id="campaign:runtime-cost-metadata"
    )
    context = _context(profile)
    proposal = _fixed_proposals(profile)["mechanism_composer"]
    entry = next(
        item
        for item in profile.entries
        if item.semantic_identity_ref
        == f"bl-icf-mechanism:{proposal.mechanism_id}"
    )
    binding = bind_search_candidate(
        profile=profile,
        proposal=proposal,
        capability_ref=entry.capability_ref,
    )
    portfolio_candidate = PortfolioCandidateV2(
        candidate_id=proposal.candidate_id,
        semantic_digest=binding.mechanism_semantics_digest,
        family_id=proposal.mechanism_axis,
        parent_id=None,
        valid_seal_probability=0.9,
        family_delta=0.0,
        parent_delta=0.0,
        information_value=0.8,
        predicted_gpu_seconds=12.0,
        age_rounds=0,
        repeat_count=0,
        lineage_risk=0.1,
        compute_pattern="pattern:runtime-cost",
        resource_admission_state=ResourceAdmissionStateV2.RESOURCE_ADMITTED,
    )
    resource_profile = {
        "schema": "test-runtime-resource.v1",
        "profile_digest": sha256_digest({"profile": proposal.candidate_id}),
        "candidate_ref": binding.capability_ref,
        "candidate_source_sha256": sha256_digest({"source": proposal.candidate_id}),
        "prediction": {
            "compute_pattern": "pattern:runtime-cost",
            "predicted_gpu_worker_seconds": 12.0,
        },
    }
    handoff = RoundCandidateHandoffV1(
        candidate_id=proposal.candidate_id,
        binding_digest=binding.digest,
        portfolio_candidate=portfolio_candidate,
        resource_profile=resource_profile,
    )
    selected_outcome = ProducerOutcome(
        producer_role=proposal.producer_role,
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=_spec(context, proposal.producer_role),
        resolution_facts={"fixture": "runtime-cost"},
        source_proposal=proposal,
    )
    acquisition = SimpleNamespace(
        route_trace=SimpleNamespace(digest=sha256_digest({"route": "runtime-cost"}))
    )

    metadata = _route_metadata_for_attempt(
        acquisition=acquisition,
        selected_outcome=selected_outcome,
        binding=binding,
        candidate_handoff=handoff,
        next_discriminative_test="retain bounded resource evidence",
        observation_seed="seed:runtime-cost",
    )

    assert metadata["selected_compute_pattern"] == "pattern:runtime-cost"
    assert metadata["selected_candidate_family_id"] == proposal.mechanism_axis
    assert metadata["selected_candidate_parent_id"] is None
    assert (
        metadata["selected_resource_admission_state"] == "RESOURCE_ADMITTED"
    )
    assert metadata["selected_resource_evidence_digest"] == sha256_digest(
        resource_profile
    )
    assert "candidate_run" not in metadata
    assert "resource_telemetry" not in metadata

    legacy_metadata = _route_metadata_for_attempt(
        acquisition=acquisition,
        selected_outcome=selected_outcome,
        binding=binding,
        next_discriminative_test="legacy route",
        observation_seed="seed:legacy",
    )
    assert "selected_compute_pattern" not in legacy_metadata
    assert "selected_resource_admission_state" not in legacy_metadata
    assert "selected_resource_evidence_digest" not in legacy_metadata
