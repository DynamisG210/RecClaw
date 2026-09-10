"""Portable identity/accounting checks; no experiment artifacts or external calls."""
from types import SimpleNamespace

from recclaw_core.research_line.runtime import (
    InnovationLaneResult,
    _innovation_attempt_proposal_digest,
)
from recclaw_core.research_line.campaign import ResearchCampaign


def attempt(proposal, bound, **extra):
    return {
        "idea_acquisition": {"selected_spec_digest": proposal},
        "spec_digest": bound,
        "candidate_root": "offline-candidate",
        **extra,
    }


def test_selection_identity_survives_parent_binding():
    assert _innovation_attempt_proposal_digest(attempt("proposal", "bound")) == "proposal"
    assert _innovation_attempt_proposal_digest({"spec_digest": "legacy"}) == "legacy"


def test_rebinding_does_not_consume_another_candidate():
    attempts = [attempt("p1", "binding1"), attempt("p1", "binding2"), attempt("p2", "binding3")]
    assert ResearchCampaign._prepared_candidate_attempt_count(attempts) == 2
    assert InnovationLaneResult.candidate_attempt_count.fget(SimpleNamespace(attempts=attempts)) == 2


def test_external_and_preimplementation_attempts_still_do_not_consume_candidates():
    attempts = [
        attempt("real", "bound", failure={"failure_class": "IMPLEMENTATION"}),
        attempt("external", "bound2", failure_scope="WORKER_TRANSIENT"),
        attempt("recovery", "bound3", failure={"failure_scope": "RECOVERY"}),
        attempt("unbound", "bound4", candidate_root=""),
    ]
    assert ResearchCampaign._prepared_candidate_attempt_count(attempts) == 1
    assert InnovationLaneResult.candidate_attempt_count.fget(SimpleNamespace(attempts=attempts)) == 1
