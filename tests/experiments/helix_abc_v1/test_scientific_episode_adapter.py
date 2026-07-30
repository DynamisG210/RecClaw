from __future__ import annotations

import ast
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.research_capability import (  # noqa: E402
    SearchMemoryWriterV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode_adapter import (  # noqa: E402
    ScientificEpisodeAdapterError,
    ScientificEpisodeAdapterReasonV1,
    project_episode_to_mechanism_belief,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (  # noqa: E402
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    EpisodeEvidenceClassV1,
    ResearchFailureClassV1,
)
from scientific_episode_fixtures import (  # noqa: E402
    comparison_identity,
    diagnostic_closure,
    digest,
    implementation_failure_receipt,
    research_episode,
    scientific_closure,
)


CANONICAL_BELIEF_DIGESTS = {
    "mechanism_negative": (
        "d5a28c46d77abce37ae9c6292d5f3fa3f174a3af52fbac5da8dc1a8d3ed77d41"
    ),
    "success": (
        "9e8c48344e992f64ebc0b3bdb4ab9b769d5600de1ee743d702b0311a3fd3f250"
    ),
}

CANONICAL_REJECTION_MATRIX = {
    "identity_drift": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "implementation_failure": (
        ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY
    ),
    "inconclusive": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "interface_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "missing_outcome": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "package_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "protocol_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "provider_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "resource_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
    "runtime_failure": ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
}


def belief_digest(belief) -> str:
    return sha256_digest(belief.to_dict())


def project(
    failure_class: ResearchFailureClassV1,
    *,
    evidence_class: EpisodeEvidenceClassV1 = (
        EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT
    ),
):
    identity = comparison_identity()
    episode = research_episode(
        identity,
        failure_class=failure_class,
        evidence_class=evidence_class,
    )
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=failure_class,
        episode=episode,
        observed_outcome_ref=episode.outcome_ref,
        observed_outcome_digest=episode.outcome_digest,
    )
    return project_episode_to_mechanism_belief(
        comparison_identity=identity,
        closure=closure,
        episode=episode,
        mechanism_axis="propagation",
    )


class ScientificEpisodeAdapterTest(unittest.TestCase):
    def test_canonical_positive_and_mechanism_negative_beliefs(self) -> None:
        beliefs = {
            "mechanism_negative": project(ResearchFailureClassV1.MECHANISM),
            "success": project(ResearchFailureClassV1.NONE),
        }
        self.assertEqual(set(beliefs), set(CANONICAL_BELIEF_DIGESTS))
        for fixture_name, belief in beliefs.items():
            with self.subTest(fixture=fixture_name):
                self.assertEqual(
                    belief_digest(belief),
                    CANONICAL_BELIEF_DIGESTS[fixture_name],
                )
                self.assertEqual(
                    belief.to_dict(),
                    project(
                        (
                            ResearchFailureClassV1.MECHANISM
                            if fixture_name == "mechanism_negative"
                            else ResearchFailureClassV1.NONE
                        )
                    ).to_dict(),
                )

        self.assertTrue(beliefs["success"].evidence_for)
        self.assertFalse(beliefs["success"].evidence_against)
        self.assertFalse(beliefs["mechanism_negative"].evidence_for)
        self.assertTrue(beliefs["mechanism_negative"].evidence_against)

    def test_projection_is_consumed_by_existing_search_memory_writer(self) -> None:
        belief = project(ResearchFailureClassV1.NONE)
        writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
        snapshot = writer.commit(
            round_index=1,
            expected_predecessor_digest=None,
            beliefs=(belief,),
            route_trace_digest=digest("adapter-route"),
            feedback_projection={
                "typed_episode_belief_digest": belief_digest(belief),
            },
        )
        self.assertEqual(snapshot.beliefs, (belief,))
        self.assertEqual(snapshot.namespace, "DEVELOPMENT_ONLY/SEARCH_MEMORY")
        self.assertIs(writer.head, snapshot)

    def test_formal_episode_is_not_upgraded_beyond_existing_memory_authority(
        self,
    ) -> None:
        belief = project(
            ResearchFailureClassV1.NONE,
            evidence_class=EpisodeEvidenceClassV1.FORMAL_EXPERIMENT,
        )
        self.assertEqual(belief.authority, "NONE")
        self.assertEqual(belief.evidence_class, "DEVELOPMENT_ONLY")
        self.assertEqual(belief.unresolved_confounds, ())

    def test_all_non_mechanism_memory_paths_fail_with_stable_reason(self) -> None:
        identity = comparison_identity()
        success_episode = research_episode(identity)
        inconclusive_episode = research_episode(
            identity,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        )
        cases = {
            "identity_drift": (
                diagnostic_closure(ResearchFailureClassV1.IDENTITY_DRIFT),
                success_episode,
            ),
            "implementation_failure": (
                diagnostic_closure(ResearchFailureClassV1.IMPLEMENTATION),
                success_episode,
            ),
            "inconclusive": (
                scientific_closure(ResearchFailureClassV1.INCONCLUSIVE),
                inconclusive_episode,
            ),
            "interface_failure": (
                diagnostic_closure(ResearchFailureClassV1.INTERFACE),
                success_episode,
            ),
            "missing_outcome": (
                diagnostic_closure(ResearchFailureClassV1.OUTCOME_MISSING),
                success_episode,
            ),
            "package_failure": (
                diagnostic_closure(ResearchFailureClassV1.PACKAGE),
                success_episode,
            ),
            "protocol_failure": (
                diagnostic_closure(ResearchFailureClassV1.PROTOCOL),
                success_episode,
            ),
            "provider_failure": (
                diagnostic_closure(ResearchFailureClassV1.PROVIDER),
                success_episode,
            ),
            "resource_failure": (
                diagnostic_closure(ResearchFailureClassV1.RESOURCE),
                success_episode,
            ),
            "runtime_failure": (
                diagnostic_closure(ResearchFailureClassV1.RUNTIME),
                success_episode,
            ),
        }
        self.assertEqual(set(cases), set(CANONICAL_REJECTION_MATRIX))
        writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
        for fixture_name, (closure, episode) in cases.items():
            with self.subTest(fixture=fixture_name):
                with self.assertRaises(ScientificEpisodeAdapterError) as raised:
                    project_episode_to_mechanism_belief(
                        comparison_identity=identity,
                        closure=closure,
                        episode=episode,
                        mechanism_axis="propagation",
                    )
                self.assertIs(
                    raised.exception.reason_code,
                    CANONICAL_REJECTION_MATRIX[fixture_name],
                )
        self.assertIsNone(writer.head)

    def test_episode_comparison_outcome_and_evidence_drift_are_rejected(
        self,
    ) -> None:
        identity = comparison_identity()
        episode = research_episode(identity)
        closure = scientific_closure(ResearchFailureClassV1.NONE)
        cases = (
            (
                "episode",
                identity,
                replace(closure, episode_digest=digest("other-episode")),
                ScientificEpisodeAdapterReasonV1.EPISODE_IDENTITY_MISMATCH,
            ),
            (
                "comparison",
                replace(identity, protocol_digest=digest("other-protocol")),
                closure,
                ScientificEpisodeAdapterReasonV1.COMPARISON_IDENTITY_MISMATCH,
            ),
            (
                "outcome",
                identity,
                replace(closure, outcome_digest=digest("other-outcome")),
                ScientificEpisodeAdapterReasonV1.OUTCOME_IDENTITY_MISMATCH,
            ),
            (
                "evidence",
                identity,
                replace(
                    closure,
                    evidence_class=EpisodeEvidenceClassV1.FORMAL_EXPERIMENT,
                ),
                ScientificEpisodeAdapterReasonV1.EVIDENCE_IDENTITY_MISMATCH,
            ),
        )
        for fixture_name, observed_identity, observed_closure, reason in cases:
            with self.subTest(fixture=fixture_name):
                with self.assertRaises(ScientificEpisodeAdapterError) as raised:
                    project_episode_to_mechanism_belief(
                        comparison_identity=observed_identity,
                        closure=observed_closure,
                        episode=episode,
                        mechanism_axis="propagation",
                    )
                self.assertIs(raised.exception.reason_code, reason)

    def test_mechanism_axis_is_normalized_and_origin_blind(self) -> None:
        identity = comparison_identity()
        episode = research_episode(identity)
        closure = scientific_closure(ResearchFailureClassV1.NONE)
        for mechanism_axis in (
            "",
            "Propagation",
            "arm_b",
            "source_b",
            "source_arm_b",
            "producer_origin",
            "message transform",
        ):
            with self.subTest(mechanism_axis=mechanism_axis):
                with self.assertRaises(ScientificEpisodeAdapterError) as raised:
                    project_episode_to_mechanism_belief(
                        comparison_identity=identity,
                        closure=closure,
                        episode=episode,
                        mechanism_axis=mechanism_axis,
                    )
                self.assertIs(
                    raised.exception.reason_code,
                    ScientificEpisodeAdapterReasonV1.MECHANISM_AXIS_INVALID,
                )

    def test_qualification_receipt_cannot_substitute_for_episode(self) -> None:
        identity = comparison_identity()
        with self.assertRaises(ScientificEpisodeAdapterError) as raised:
            project_episode_to_mechanism_belief(
                comparison_identity=identity,
                closure=diagnostic_closure(
                    ResearchFailureClassV1.IMPLEMENTATION
                ),
                episode=implementation_failure_receipt(),
                mechanism_axis="propagation",
            )
        self.assertIs(
            raised.exception.reason_code,
            ScientificEpisodeAdapterReasonV1.INVALID_INPUT_TYPE,
        )

    def test_projection_has_exact_existing_fields_and_no_origin_metadata(
        self,
    ) -> None:
        belief = project(ResearchFailureClassV1.NONE)
        self.assertEqual(
            tuple(belief.to_dict()),
            (
                "competing_hypotheses",
                "evidence_against",
                "evidence_for",
                "hypothesis_id",
                "mechanism_axis",
                "next_discriminative_test",
                "predicted_outcome_signature",
                "unresolved_confounds",
            ),
        )
        serialized = str(belief.to_dict()).lower()
        for forbidden in (
            "origin",
            "producer",
            "source_arm",
            "owner_arm",
            "arm_instance",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_adapter_import_graph_has_no_writer_guard_meta_or_frontier(self) -> None:
        source = (
            SRC
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "scientific_episode_adapter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        joined = " ".join(imports).lower()
        for forbidden in ("guard", "meta", "writer", "frontier"):
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
