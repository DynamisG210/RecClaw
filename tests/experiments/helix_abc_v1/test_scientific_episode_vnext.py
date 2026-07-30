from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (  # noqa: E402
    EpisodeClosureStatusV1,
    EpisodeMemoryLaneV1,
    ScientificEpisodeClosureV1,
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    EpisodeEvidenceClassV1,
    QualificationFailureClassV1,
    ResearchFailureClassV1,
    VNextContractError,
)
from scientific_episode_fixtures import (  # noqa: E402
    canonical_closure_fixtures,
    comparison_identity,
    diagnostic_closure,
    digest,
    identity_drifted_episode,
    implementation_failure_receipt,
    passing_qualification_receipt,
    research_episode,
    scientific_closure,
)


STAGE_FIELDS = (
    "identity_result",
    "protocol_result",
    "package_result",
    "interface_result",
    "execution_result",
    "outcome_result",
    "comparator_result",
    "evidence_result",
)

CANONICAL_FIXTURE_DIGESTS = {
    "identity_drift": (
        "500d0d4ca137a5573505547c061b917a18aa19d70b01a9d7059b8a586aee5eea"
    ),
    "implementation_failure": (
        "1626b36b537e555138dca66ecf81a7d81def08b1afd99fa8c047e88845025962"
    ),
    "inconclusive": (
        "bb4d542d3247e6ed0b68bed359f18614e1ba99812a8d4b54cfa9132ec7ad8988"
    ),
    "interface_failure": (
        "dfb7472fc06af8a5c0f333eaa09233cc218cdc5b8849dd6aba0c2443b8e4d20c"
    ),
    "mechanism_negative": (
        "41830a61a88b3586acdfef25fca125464b466d734e5d52e7d7d88fb7948f36e0"
    ),
    "missing_outcome": (
        "b9cd6a4deeaaafcc6fe8f4f21093712566e3455ed357abe24fdd99aa9107d547"
    ),
    "package_failure": (
        "4038975935e871180266f1eaf5fc56835483c52e4c3bc2cff60d529d0b58c913"
    ),
    "protocol_failure": (
        "14a36d4f84706d705c09eaf2ff6b417abdc8915b9a4ef9bf35d135a789a0b035"
    ),
    "provider_failure": (
        "09d4898ec4a50735eba7750936fa1d522eca6c828ffd1fab58a5113944895241"
    ),
    "resource_failure": (
        "e1942f1d3a3153015e2e7317921f2239bbd44e2f45d8ee394f70aba8b759a1c4"
    ),
    "runtime_failure": (
        "3f4f679740a5c51fde97d1d8d0f76414055db17ae206e8278ee45e66060e13d1"
    ),
    "success": (
        "ddb2d594369b740e389ebc3fdb2a973aa3284ca34e87d3ba53c1dc262aa2417e"
    ),
}


class ScientificEpisodeVNextTest(unittest.TestCase):
    def test_canonical_fixture_matrix_is_complete_and_deterministic(self) -> None:
        first = canonical_closure_fixtures()
        second = canonical_closure_fixtures()
        self.assertEqual(
            tuple(first),
            (
                "identity_drift",
                "implementation_failure",
                "inconclusive",
                "interface_failure",
                "mechanism_negative",
                "missing_outcome",
                "package_failure",
                "protocol_failure",
                "provider_failure",
                "resource_failure",
                "runtime_failure",
                "success",
            ),
        )
        self.assertEqual(set(first), set(CANONICAL_FIXTURE_DIGESTS))
        for fixture_name, closure in first.items():
            with self.subTest(fixture=fixture_name):
                self.assertEqual(
                    closure.canonical_bytes(),
                    canonical_json_bytes(closure.canonical_dict()),
                )
                self.assertEqual(
                    closure.canonical_bytes(),
                    second[fixture_name].canonical_bytes(),
                )
                self.assertEqual(
                    closure.digest,
                    CANONICAL_FIXTURE_DIGESTS[fixture_name],
                )

    def test_only_complete_scientific_comparisons_enter_mechanism_memory(
        self,
    ) -> None:
        success = scientific_closure(ResearchFailureClassV1.NONE)
        mechanism_negative = scientific_closure(
            ResearchFailureClassV1.MECHANISM
        )
        for closure in (success, mechanism_negative):
            with self.subTest(failure_class=closure.failure_class.value):
                self.assertIs(
                    closure.memory_lane,
                    EpisodeMemoryLaneV1.MECHANISM_MEMORY,
                )
                self.assertTrue(closure.mechanism_memory_allowed)
                self.assertFalse(closure.engineering_diagnostic_allowed)
                self.assertTrue(
                    all(
                        getattr(closure, name) is EpisodeClosureStatusV1.PASS
                        for name in STAGE_FIELDS
                    )
                )

        identity = comparison_identity()
        inconclusive_episode = research_episode(
            identity,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        )
        inconclusive = close_scientific_episode(
            comparison_identity=identity,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            episode=inconclusive_episode,
            observed_outcome_ref=inconclusive_episode.outcome_ref,
            observed_outcome_digest=inconclusive_episode.outcome_digest,
        )
        self.assertIs(inconclusive.memory_lane, EpisodeMemoryLaneV1.NONE)
        self.assertFalse(inconclusive.mechanism_memory_allowed)
        self.assertFalse(inconclusive.engineering_diagnostic_allowed)

    def test_diagnostic_failure_matrix_routes_only_to_engineering_lane(self) -> None:
        failures = (
            ResearchFailureClassV1.IMPLEMENTATION,
            ResearchFailureClassV1.INTERFACE,
            ResearchFailureClassV1.PACKAGE,
            ResearchFailureClassV1.PROTOCOL,
            ResearchFailureClassV1.RUNTIME,
            ResearchFailureClassV1.RESOURCE,
            ResearchFailureClassV1.PROVIDER,
            ResearchFailureClassV1.OUTCOME_MISSING,
            ResearchFailureClassV1.IDENTITY_DRIFT,
        )
        for failure_class in failures:
            with self.subTest(failure_class=failure_class.value):
                closure = diagnostic_closure(failure_class)
                self.assertIs(
                    closure.memory_lane,
                    EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC,
                )
                self.assertFalse(closure.mechanism_memory_allowed)
                self.assertTrue(closure.engineering_diagnostic_allowed)
                self.assertIsNone(closure.episode_ref)
                self.assertIsNone(closure.episode_digest)
                self.assertIsNone(closure.outcome_ref)
                self.assertIsNone(closure.outcome_digest)
                results = tuple(getattr(closure, name) for name in STAGE_FIELDS)
                failure_index = results.index(EpisodeClosureStatusV1.FAIL)
                self.assertTrue(
                    all(
                        item is EpisodeClosureStatusV1.PASS
                        for item in results[:failure_index]
                    )
                )
                self.assertTrue(
                    all(
                        item is EpisodeClosureStatusV1.NOT_RUN
                        for item in results[failure_index + 1 :]
                    )
                )

    def test_qualification_taxonomy_is_development_only_and_non_mechanistic(
        self,
    ) -> None:
        values = {item.value for item in QualificationFailureClassV1}
        self.assertIn("PACKAGE", values)
        self.assertIn("PROVIDER", values)
        self.assertNotIn("MECHANISM", values)

    def test_typed_episode_rejects_every_non_scientific_failure(self) -> None:
        identity = comparison_identity()
        episode = research_episode(identity)
        for failure_class in (
            ResearchFailureClassV1.IMPLEMENTATION,
            ResearchFailureClassV1.INTERFACE,
            ResearchFailureClassV1.PACKAGE,
            ResearchFailureClassV1.PROTOCOL,
            ResearchFailureClassV1.RUNTIME,
            ResearchFailureClassV1.RESOURCE,
            ResearchFailureClassV1.PROVIDER,
            ResearchFailureClassV1.OUTCOME_MISSING,
            ResearchFailureClassV1.IDENTITY_DRIFT,
        ):
            with self.subTest(failure_class=failure_class.value):
                with self.assertRaisesRegex(
                    VNextContractError,
                    "only an executed scientific comparison",
                ):
                    replace(
                        episode,
                        evidence_class=EpisodeEvidenceClassV1.ENGINEERING_ONLY,
                        failure_class=failure_class,
                        mechanism_interpretation="NOT_ADJUDICATED",
                    )

    def test_identity_drift_is_rejected_before_memory_permission(self) -> None:
        episode = identity_drifted_episode()
        with self.assertRaisesRegex(
            VNextContractError,
            "experiment_binding_digest",
        ):
            close_scientific_episode(
                comparison_identity=comparison_identity(),
                failure_class=ResearchFailureClassV1.NONE,
                episode=episode,
                observed_outcome_ref=episode.outcome_ref,
                observed_outcome_digest=episode.outcome_digest,
            )

    def test_qualification_reference_is_optional_but_never_scientific_evidence(
        self,
    ) -> None:
        identity = comparison_identity()
        passing = passing_qualification_receipt()
        episode = replace(
            research_episode(identity),
            qualification_receipt_ref=passing.receipt_id,
            qualification_receipt_digest=passing.digest,
        )
        with self.assertRaisesRegex(
            VNextContractError,
            "must be supplied",
        ):
            close_scientific_episode(
                comparison_identity=identity,
                failure_class=ResearchFailureClassV1.NONE,
                episode=episode,
                observed_outcome_ref=episode.outcome_ref,
                observed_outcome_digest=episode.outcome_digest,
            )
        closure = close_scientific_episode(
            comparison_identity=identity,
            failure_class=ResearchFailureClassV1.NONE,
            episode=episode,
            observed_outcome_ref=episode.outcome_ref,
            observed_outcome_digest=episode.outcome_digest,
            qualification_receipt=passing,
        )
        self.assertTrue(closure.mechanism_memory_allowed)
        self.assertEqual(closure.qualification_receipt_ref, passing.receipt_id)

        failed = implementation_failure_receipt()
        failed_episode = replace(
            research_episode(identity),
            qualification_receipt_ref=failed.receipt_id,
            qualification_receipt_digest=failed.digest,
        )
        with self.assertRaisesRegex(
            VNextContractError,
            "passing one-epoch",
        ):
            close_scientific_episode(
                comparison_identity=identity,
                failure_class=ResearchFailureClassV1.NONE,
                episode=failed_episode,
                observed_outcome_ref=failed_episode.outcome_ref,
                observed_outcome_digest=failed_episode.outcome_digest,
                qualification_receipt=failed,
            )

    def test_outcome_identity_drift_and_missing_outcome_are_rejected(self) -> None:
        identity = comparison_identity()
        episode = research_episode(identity)
        for outcome_ref, outcome_digest in (
            (None, None),
            (episode.outcome_ref, digest("different-outcome")),
        ):
            with self.subTest(
                outcome_ref=outcome_ref,
                outcome_digest=outcome_digest,
            ):
                with self.assertRaisesRegex(
                    VNextContractError,
                    "observed outcome identity drift",
                ):
                    close_scientific_episode(
                        comparison_identity=identity,
                        failure_class=ResearchFailureClassV1.NONE,
                        episode=episode,
                        observed_outcome_ref=outcome_ref,
                        observed_outcome_digest=outcome_digest,
                    )

    def test_diagnostic_closure_rejects_episode_or_fabricated_outcome(self) -> None:
        identity = comparison_identity()
        episode = research_episode(identity)
        common = {
            "comparison_identity": identity,
            "failure_class": ResearchFailureClassV1.RUNTIME,
            "failure_detail_ref": "diagnostic:runtime:v1",
            "failure_detail_digest": digest("runtime-detail"),
        }
        with self.assertRaisesRegex(
            VNextContractError,
            "cannot create TypedResearchEpisodeV1",
        ):
            close_scientific_episode(
                **common,
                episode=episode,
                observed_outcome_ref=None,
                observed_outcome_digest=None,
            )
        with self.assertRaisesRegex(
            VNextContractError,
            "preserve the outcome as missing",
        ):
            close_scientific_episode(
                **common,
                episode=None,
                observed_outcome_ref=episode.outcome_ref,
                observed_outcome_digest=episode.outcome_digest,
            )

    def test_closure_contract_rejects_non_structured_stage_sequences(self) -> None:
        closure = diagnostic_closure(ResearchFailureClassV1.RUNTIME)
        with self.assertRaisesRegex(
            VNextContractError,
            "PASS prefix, one FAIL, and later NOT_RUN",
        ):
            replace(
                closure,
                outcome_result=EpisodeClosureStatusV1.PASS,
            )

    def test_direct_contract_construction_cannot_grant_memory_permission(self) -> None:
        closure = diagnostic_closure(ResearchFailureClassV1.PROTOCOL)
        with self.assertRaisesRegex(
            VNextContractError,
            "memory permission flags",
        ):
            replace(
                closure,
                mechanism_memory_allowed=True,
            )
        with self.assertRaisesRegex(
            VNextContractError,
            "memory lane",
        ):
            replace(
                closure,
                memory_lane=EpisodeMemoryLaneV1.MECHANISM_MEMORY,
            )

    def test_direct_contract_rejects_scientific_evidence_class_drift(self) -> None:
        closure = scientific_closure(ResearchFailureClassV1.NONE)
        with self.assertRaisesRegex(
            VNextContractError,
            "terminal and evidence class disagree",
        ):
            replace(
                closure,
                evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
                memory_lane=EpisodeMemoryLaneV1.NONE,
                mechanism_memory_allowed=False,
            )

    def test_closure_type_is_not_a_second_episode_contract(self) -> None:
        closure = scientific_closure(ResearchFailureClassV1.NONE)
        self.assertIsInstance(closure, ScientificEpisodeClosureV1)
        self.assertNotIn("hypothesis", closure.to_dict())
        self.assertNotIn("mechanism_interpretation", closure.to_dict())


if __name__ == "__main__":
    unittest.main()
