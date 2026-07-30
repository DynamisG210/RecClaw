from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (  # noqa: E402
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (  # noqa: E402
    HIGH_CHANGE_DIMENSIONS,
    frozen_search_bindings,
    frozen_search_resolver_environment,
    project_candidate_proposal_v4,
    project_open_producer_draft,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    CandidateProposalV4,
    DiscriminativeExperimentPlanV1,
    DiscoveryCreditV1,
    DISCOVERY_PRODUCERS,
    MatchedControlPlanV1,
    ProposalIntentV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    CapabilityResolutionResultV1,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "a0_open_spec_cases_v1.json"
)


def fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def candidate_proposal(
    *,
    mechanism: object,
    role: str,
    index: int,
    protocol_digest: str,
) -> CandidateProposalV4:
    candidate_id = f"cand-a0-{index:03d}"
    control = MatchedControlPlanV1(
        mechanism_question_digest=sha256_digest(
            {"a0-mechanism": mechanism.mechanism_id}
        ),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest=protocol_digest,
        changed_axis=mechanism.mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    discriminative = (
        DiscriminativeExperimentPlanV1(
            competing_hypotheses=(
                "The mechanism causes the signature.",
                "The signature is explained by the matched control.",
            ),
            predicted_outcome_signature="A role-preserving predicted signature.",
            primary_candidate=candidate_id,
            matched_control_plan=control,
            falsifier="The matched control reproduces the signature.",
            next_decision_rule="Prefer the explanation with the matched signature.",
        )
        if role == "falsification_designer"
        else None
    )
    utility = SearchUtilityFeaturesV1(
        runnable_probability=0.8,
        useful_signal=0.7,
        frontier_potential=0.6,
        information_gain=0.7,
        cost=0.2,
        blocker_risk=0.1,
    )
    evidence = RouterFeatureEvidenceV1(
        compile_valid=True,
        handler_available=True,
        materializer_available=True,
        blocker_rate=0.0,
        semantic_duplicate=False,
        parent_available=True,
        mechanism_depth=len(mechanism.operator_ids),
        estimated_cost=0.2,
        llm_diagnostic=utility,
    )
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id=f"producer-a0-{role}",
        producer_role=role,
        proposal_intent=(
            ProposalIntentV1.FALSIFICATION
            if role == "falsification_designer"
            else ProposalIntentV1.DISCOVERY
        ),
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        candidate_label=f"A0 projection for {mechanism.mechanism_id}",
        mechanism_hypothesis=f"Hypothesis for {mechanism.mechanism_id}.",
        competing_hypothesis=f"Competing explanation for {mechanism.mechanism_id}.",
        predicted_outcome_signature="A role-preserving predicted signature.",
        failure_mode="The predicted signature is absent.",
        utility_features=utility,
        feature_evidence=evidence,
        matched_control_plan=control,
        discriminative_plan=discriminative,
        parent_candidate_id=None,
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


class A0CanonicalFixturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixture()
        self.bindings = self.fixture["bindings"]
        self.environment = self.fixture["environment"]

    def _case_draft(self, case: dict[str, object]) -> dict[str, object]:
        if "draft" in case:
            return copy.deepcopy(case["draft"])
        return copy.deepcopy(
            self.fixture["producer_drafts"][case["draft_index"]]
        )

    def test_canonical_fixture_covers_exactly_five_resolution_states(self) -> None:
        observed = {}
        for case in self.fixture["resolution_cases"]:
            spec, facts = project_open_producer_draft(
                self._case_draft(case),
                bindings=self.bindings,
            )
            resolution = resolve_capability(
                spec,
                resolution_facts=facts,
                environment=self.environment,
            )
            observed[case["fixture_id"]] = resolution.resolution.value
            self.assertEqual(
                resolution.resolution.value,
                case["expected_resolution"],
            )
        self.assertEqual(
            set(observed.values()),
            {item.value for item in CapabilityResolutionResultV1},
        )

    def test_all_four_existing_roles_project_without_semantic_relabeling(self) -> None:
        observed_roles = set()
        for draft in self.fixture["producer_drafts"]:
            spec, facts = project_open_producer_draft(
                draft,
                bindings=self.bindings,
            )
            observed_roles.add(spec.producer_role)
            self.assertEqual(spec.hypothesis, draft["hypothesis"])
            self.assertEqual(
                spec.competing_explanation,
                draft["competing_explanation"],
            )
            self.assertEqual(spec.falsifier, draft["falsifier"])
            self.assertEqual(
                spec.matched_control_requirement,
                draft["matched_control_requirement"],
            )
            resolution = resolve_capability(
                spec,
                resolution_facts=facts,
                environment=self.environment,
            )
            self.assertIs(
                resolution.resolution,
                CapabilityResolutionResultV1.INNOVATION_REQUIRED,
            )
        self.assertEqual(observed_roles, set(DISCOVERY_PRODUCERS))

    def test_input_order_does_not_change_projection_or_resolution(self) -> None:
        draft = copy.deepcopy(self.fixture["producer_drafts"][2])
        reordered_draft = copy.deepcopy(draft)
        for field in (
            "implementation_requirements",
            "expected_evidence",
            "compatibility_requirements",
        ):
            reordered_draft[field] = list(reversed(reordered_draft[field]))
        for field in (
            "capability_diff",
            "high_change_dimensions",
            "required_dependencies",
        ):
            reordered_draft["resolution_facts"][field] = list(
                reversed(reordered_draft["resolution_facts"][field])
            )
        reordered_bindings = {
            key: copy.deepcopy(self.bindings[key])
            for key in reversed(tuple(self.bindings))
        }
        reordered_environment = {
            key: copy.deepcopy(self.environment[key])
            for key in reversed(tuple(self.environment))
        }
        reordered_environment["protocol_requirements"].reverse()
        reordered_environment["available_dependencies"].reverse()
        reordered_environment["current_capabilities"].reverse()

        first_spec, first_facts = project_open_producer_draft(
            draft,
            bindings=self.bindings,
        )
        second_spec, second_facts = project_open_producer_draft(
            reordered_draft,
            bindings=reordered_bindings,
        )
        first = resolve_capability(
            first_spec,
            resolution_facts=first_facts,
            environment=self.environment,
        )
        second = resolve_capability(
            second_spec,
            resolution_facts=second_facts,
            environment=reordered_environment,
        )
        self.assertEqual(first_spec.digest, second_spec.digest)
        self.assertEqual(first_facts, second_facts)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())

    def test_invalid_raw_contract_and_invalid_resolution_facts_are_typed(self) -> None:
        draft = self._case_draft(self.fixture["resolution_cases"][1])
        spec, facts = project_open_producer_draft(
            draft,
            bindings=self.bindings,
        )
        invalid_spec = spec.to_dict()
        invalid_spec["falsifier"] = ""
        invalid_contract = resolve_capability(
            invalid_spec,
            resolution_facts=facts,
            environment=self.environment,
        )
        self.assertIs(
            invalid_contract.resolution,
            CapabilityResolutionResultV1.INVALID_SPEC,
        )
        self.assertEqual(
            invalid_contract.reason_codes,
            ("OPEN_SPEC_CONTRACT_INVALID",),
        )
        invalid_facts = dict(facts)
        invalid_facts.pop("required_budget")
        invalid_resolution_facts = resolve_capability(
            spec,
            resolution_facts=invalid_facts,
            environment=self.environment,
        )
        self.assertIs(
            invalid_resolution_facts.resolution,
            CapabilityResolutionResultV1.INVALID_SPEC,
        )

    def test_protocol_change_precedes_implementation_and_never_executes(self) -> None:
        base = self._case_draft(self.fixture["resolution_cases"][1])
        for requirement in (
            "time split evaluation",
            "online feedback",
            "new image modality",
        ):
            draft = copy.deepcopy(base)
            draft["compatibility_requirements"].append(requirement)
            draft["resolution_facts"]["required_dependencies"] = [
                "unavailable-after-protocol-change"
            ]
            spec, facts = project_open_producer_draft(
                draft,
                bindings=self.bindings,
            )
            resolution = resolve_capability(
                spec,
                resolution_facts=facts,
                environment=self.environment,
            )
            self.assertIs(
                resolution.resolution,
                CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE,
            )
            self.assertFalse(resolution.protocol_compatible)
            self.assertFalse(resolution.current_profile_match)

    def test_outside_profile_never_falls_back_to_nearest_catalog_entry(self) -> None:
        draft = self._case_draft(self.fixture["resolution_cases"][1])
        draft["resolution_facts"][
            "requested_current_semantics_digest"
        ] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        spec, facts = project_open_producer_draft(
            draft,
            bindings=self.bindings,
        )
        resolution = resolve_capability(
            spec,
            resolution_facts=facts,
            environment=self.environment,
        )
        self.assertIs(
            resolution.resolution,
            CapabilityResolutionResultV1.INNOVATION_REQUIRED,
        )
        self.assertTrue(resolution.no_silent_fallback)
        self.assertFalse(resolution.catalog_fallback_used)
        self.assertIsNone(resolution.resolved_current_capability_ref)
        self.assertIn("OUTSIDE_CURRENT_PROFILE", resolution.reason_codes)

    def test_innovation_is_pure_and_next_fresh_campaign_only(self) -> None:
        draft = self._case_draft(self.fixture["resolution_cases"][1])
        spec, facts = project_open_producer_draft(
            draft,
            bindings=self.bindings,
        )
        environment_before = canonical_json_bytes(self.environment)
        spec_before = spec.canonical_bytes()
        first = resolve_capability(
            spec,
            resolution_facts=facts,
            environment=self.environment,
        )
        second = resolve_capability(
            spec,
            resolution_facts=facts,
            environment=self.environment,
        )
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(
            environment_before,
            canonical_json_bytes(self.environment),
        )
        self.assertEqual(spec_before, spec.canonical_bytes())
        self.assertEqual(
            first.current_profile_digest,
            self.environment["current_profile_digest"],
        )
        self.assertIn("NEXT_FRESH_CAMPAIGN_ONLY", first.reason_codes)

    def test_high_change_dimensions_cover_frozen_mechanism_boundaries(self) -> None:
        self.assertTrue(
            {
                "CORE_REPRESENTATION",
                "CORE_OBJECTIVE",
                "CORE_RELATION",
                "PROPAGATION_MECHANISM",
                "TRAINING_PROCEDURE",
                "MODEL_STRUCTURE",
            }.issubset(HIGH_CHANGE_DIMENSIONS)
        )


class A0FrozenSearchProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = frozen_search_resolver_environment(
            available_dependencies=("recbole-runtime",),
            budget_limits={"implementation_units": 1},
        )
        cls.bindings = frozen_search_bindings(
            context_ref="context:a0-frozen-66",
            context_digest=sha256_digest({"fixture": "a0-frozen-66-context"}),
        )

    def test_all_66_frozen_candidates_are_exact_search_ready(self) -> None:
        mechanisms = executable_mechanisms()
        self.assertEqual(len(mechanisms), 66)
        observed_roles = set()
        for index, mechanism in enumerate(mechanisms):
            role = DISCOVERY_PRODUCERS[index % len(DISCOVERY_PRODUCERS)]
            proposal = candidate_proposal(
                mechanism=mechanism,
                role=role,
                index=index,
                protocol_digest=self.bindings["protocol_digest"],
            )
            spec, facts = project_candidate_proposal_v4(
                proposal,
                bindings=self.bindings,
                required_dependencies=("recbole-runtime",),
                required_budget={"implementation_units": 1},
            )
            resolution = resolve_capability(
                spec,
                resolution_facts=facts,
                environment=self.environment,
            )
            observed_roles.add(spec.producer_role)
            self.assertEqual(
                spec.hypothesis,
                proposal.mechanism_hypothesis,
            )
            self.assertEqual(
                spec.competing_explanation,
                proposal.competing_hypothesis,
            )
            self.assertIs(
                resolution.resolution,
                CapabilityResolutionResultV1.SEARCH_READY,
            )
            self.assertTrue(resolution.current_profile_match)
            self.assertTrue(resolution.no_silent_fallback)
            self.assertFalse(resolution.catalog_fallback_used)
            self.assertEqual(resolution.capability_diff, ())
            self.assertTrue(
                resolution.resolved_current_capability_ref.endswith(
                    f":{mechanism.mechanism_id}"
                )
            )
        self.assertEqual(observed_roles, set(DISCOVERY_PRODUCERS))

    def test_budget_excess_is_unsupported_without_profile_substitution(self) -> None:
        draft = fixture()["producer_drafts"][0]
        spec, facts = project_open_producer_draft(
            draft,
            bindings=fixture()["bindings"],
        )
        facts["required_budget"]["implementation_units"] = 3
        resolution = resolve_capability(
            spec,
            resolution_facts=facts,
            environment=fixture()["environment"],
        )
        self.assertIs(
            resolution.resolution,
            CapabilityResolutionResultV1.UNSUPPORTED,
        )
        self.assertFalse(resolution.budget_compatible)
        self.assertIsNone(resolution.resolved_current_capability_ref)


if __name__ == "__main__":
    unittest.main()
