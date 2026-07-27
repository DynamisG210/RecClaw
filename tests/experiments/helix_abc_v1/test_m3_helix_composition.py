from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.helix.composition import SameSlateHelixSelectorV1  # noqa: E402
from recclaw_core.helix.contracts import (  # noqa: E402
    CandidateEnvelope,
    CompactFeedback,
    GuardContext,
    PortStatus,
    RawResultEnvelope,
)
from recclaw_core.helix.fusion import (  # noqa: E402
    DeterministicHelixFusionV1,
    HelixFusionBridgeV1,
)
from recclaw_core.helix.guard_adapter import EvidenceGuardPortV1  # noqa: E402
from recclaw_core.helix.ledger import (  # noqa: E402
    EvidenceGuardLedgerWriterV1,
    GuardLedgerError,
)
from recclaw_core.helix.ports import NullEvidencePortV1  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)


def protocol() -> dict[str, object]:
    return {
        "protocol_id": "PROTO-ML1M-FULL-001",
        "profile_family": "OFFLINE_TOPN",
        "dataset": "ml-1m",
        "dataset_snapshot": "ml-1m-snapshot-001",
        "split": {"strategy": "random_user_holdout", "ratio": [0.8, 0.1, 0.1]},
        "training_sampling": {"mode": "uniform_negative"},
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {"seen_items": "exclude"},
        "metric": {"name": "ndcg", "cutoff": 10},
        "training_procedure": {"optimizer": "adam", "max_epochs": 300},
    }


def context() -> GuardContext:
    return GuardContext(
        claim={
            "claim_id": "CLAIM-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "claim_kind": "LOCAL_IMPROVEMENT",
            "target_model": "CandidateModel",
            "comparator": "LightGCN",
            "metric": "ndcg",
            "required_seed_count": 3,
            "scope": {"dataset": "ml-1m"},
        },
        protocol=protocol(),
        current_evidence={
            "snapshot_id": "EMPTY-SNAPSHOT",
            "claim_id": "CLAIM-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "observation_ids": [],
        },
    )


def candidate(
    candidate_id: str = "cand-001",
    *,
    planned: dict[str, object] | None = None,
    common_plan_digest: str | None = None,
    arm_id: str = "opaque-c",
    seed_id: str = "2026",
) -> CandidateEnvelope:
    return CandidateEnvelope(
        candidate_id=candidate_id,
        candidate_semantic_digest=hashlib.sha256(
            f"semantic:{candidate_id}".encode()
        ).hexdigest(),
        opaque_arm_instance_id=arm_id,
        common_status="COMMON_PASS",
        mechanism_program_digest=hashlib.sha256(
            f"program:{candidate_id}".encode()
        ).hexdigest(),
        common_plan_digest=common_plan_digest
        or hashlib.sha256(f"plan:{candidate_id}".encode()).hexdigest(),
        action_family="RUN_OFFLINE_TOPN",
        planned_protocol=planned or protocol(),
        target_model="CandidateModel",
        comparator="LightGCN",
        seed_ids=(seed_id,),
        purpose="development comparison",
    )


def result(
    candidate_id: str = "cand-001",
    *,
    observed: dict[str, object] | None = None,
    raw_identity: str = "raw-001",
    arm_id: str = "opaque-c",
    seed_id: str = "2026",
    ndcg: float = 0.27,
) -> RawResultEnvelope:
    return RawResultEnvelope(
        candidate_id=candidate_id,
        opaque_arm_instance_id=arm_id,
        raw_result_digest=hashlib.sha256(raw_identity.encode()).hexdigest(),
        common_result_closure_digest=hashlib.sha256(
            f"closure:{raw_identity}".encode()
        ).hexdigest(),
        observed_protocol=observed or protocol(),
        target_model="CandidateModel",
        comparator="LightGCN",
        seed_runs=(
            {
                "seed_id": seed_id,
                "run_id": f"run-{raw_identity}",
                "artifact_sha256": hashlib.sha256(
                    f"artifact:{raw_identity}".encode()
                ).hexdigest(),
            },
        ),
        observation_kind="METRIC_EVALUATION",
        run_status="SUCCESS",
        artifact_identity_status="EXACT",
        normalized_metrics={"ndcg": ndcg},
    )


class M3HelixCompositionTest(unittest.TestCase):
    def test_selected_guard_core_is_exact_byte_copy(self) -> None:
        selected = SRC / "recclaw_evidence_guard" / "core_v1.py"
        self.assertEqual(
            hashlib.sha256(selected.read_bytes()).hexdigest(),
            "d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af",
        )

    def test_only_guard_adapter_is_shared_direct_import_adjacency(self) -> None:
        offenders = []
        for path in (SRC / "recclaw_core").rglob("*.py"):
            if "evidence_guard" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = {
                node.module if isinstance(node, ast.ImportFrom) else alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            if any(name.startswith("recclaw_evidence_guard") for name in imports):
                offenders.append(path.relative_to(SRC).as_posix())
        self.assertEqual(offenders, ["recclaw_core/helix/guard_adapter.py"])

    def test_research_and_common_runtime_bytes_remain_frozen(self) -> None:
        m2_path = (
            ROOT
            / "docs/research_line/m2/RESEARCH_LINE_STANDALONE_READINESS_V1.json"
        )
        m2 = json.loads(m2_path.read_text())
        successor = json.loads(
            (
                ROOT
                / "docs/research_line/readiness/NON_META_RESEARCH_LINE_RELEASE_V2.json"
            ).read_text()
        )
        self.assertEqual(
            successor["predecessor"]["artifact_path"],
            m2_path.relative_to(ROOT).as_posix(),
        )
        self.assertEqual(
            successor["predecessor"]["artifact_sha256"],
            hashlib.sha256(m2_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            successor["predecessor"]["content_digest"],
            m2["content_digest"],
        )
        audit_report = successor["audit_report"]
        self.assertEqual(
            hashlib.sha256((ROOT / audit_report["artifact_path"]).read_bytes()).hexdigest(),
            audit_report["artifact_sha256"],
        )
        for path, digest in successor["preserved_historical_artifacts"]:
            self.assertEqual(
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                digest,
            )
        successor_preimage = copy.deepcopy(successor)
        successor_preimage.pop("content_digest")
        self.assertEqual(
            sha256_digest(successor_preimage),
            successor["content_digest"],
        )
        campaign_contract = json.loads(
            (
                ROOT
                / "docs/research_line/readiness/"
                "CAMPAIGN_PILOT_V12_FROZEN_CONTRACT.json"
            ).read_text()
        )
        current_sources = campaign_contract["source"]["files"]
        for path, digest in m2["component_digests"]:
            current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if current != digest and not path.startswith("tests/"):
                self.assertEqual(current_sources[path], current)
        runtime_repair = json.loads(
            (
                ROOT
                / "docs/research_line/readiness/"
                "RESEARCH_LINE_RUNTIME_REPAIR_RELEASE_V3.json"
            ).read_text()
        )
        self.assertEqual(
            runtime_repair["predecessor"]["artifact_path"],
            (
                "docs/research_line/readiness/"
                "NON_META_RESEARCH_LINE_RELEASE_V2.json"
            ),
        )
        self.assertEqual(
            runtime_repair["predecessor"]["artifact_sha256"],
            hashlib.sha256(
                (
                    ROOT
                    / runtime_repair["predecessor"]["artifact_path"]
                ).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            runtime_repair["predecessor"]["content_digest"],
            successor["content_digest"],
        )
        repair_preimage = copy.deepcopy(runtime_repair)
        repair_preimage.pop("content_digest")
        self.assertEqual(
            sha256_digest(repair_preimage),
            runtime_repair["content_digest"],
        )
        for path, _digest in successor["component_digests"]:
            current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if current != _digest and not path.startswith("tests/"):
                self.assertEqual(current_sources[path], current)
        m1 = json.loads(
            (
                ROOT
                / "docs/research_line/m1/COMMON_EXECUTION_GUARD_RELEASE_PROJECTION_V1.json"
            ).read_text()
        )
        self.assertEqual(
            m1["source_manifest_digest"],
            "038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd",
        )

    def test_null_port_is_not_adjudicated_and_creates_no_guard_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adjudication = NullEvidencePortV1().pre_run(candidate())
            self.assertEqual(adjudication.status, PortStatus.NOT_ADJUDICATED)
            self.assertFalse(root.exists() and any(root.iterdir()))

    def test_same_fusion_selects_null_first_candidate_without_pretending_allow(self) -> None:
        fusion = DeterministicHelixFusionV1()
        selection = SameSlateHelixSelectorV1(fusion).select(
            (candidate("cand-a"), candidate("cand-b")), NullEvidencePortV1()
        )
        self.assertEqual(selection.selected_candidate.candidate_id, "cand-a")
        self.assertEqual(selection.terminal_status, "SELECTED_NOT_ADJUDICATED")
        self.assertEqual(selection.producer_refresh_count, 0)
        self.assertEqual(selection.extra_proposal_count, 0)

    def test_c_pre_block_advances_only_within_frozen_slate(self) -> None:
        flipped = protocol()
        flipped["protocol_id"] = "PROTO-SAMPLED"
        flipped["evaluation_candidate_universe"] = {"mode": "sampled", "size": 100}
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit")
            port = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            original = (candidate("cand-block", planned=flipped), candidate("cand-allow"))
            before = tuple(item.digest for item in original)
            selection = SameSlateHelixSelectorV1(
                DeterministicHelixFusionV1()
            ).select(original, port)
            self.assertEqual(selection.selected_candidate.candidate_id, "cand-allow")
            self.assertEqual(selection.inspected_candidate_ids, ("cand-block", "cand-allow"))
            self.assertEqual(tuple(item.digest for item in original), before)
            self.assertEqual(selection.producer_refresh_count, 0)
            self.assertEqual(selection.extra_proposal_count, 0)
            self.assertEqual(ledger.count(), 2)

    def test_post_exact_result_produces_exact_eight_field_compact_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit")
            port = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            item = candidate()
            self.assertEqual(port.pre_run(item).status, PortStatus.ALLOW)
            post = port.post_run(result())
            disposition = DeterministicHelixFusionV1().fuse(post)
            feedback = disposition.compact_feedback
            self.assertIsNotNone(feedback)
            self.assertEqual(len(fields(feedback)), 8)
            self.assertEqual(
                feedback.evidence_use, "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL"
            )
            instruction = HelixFusionBridgeV1().map(feedback)
            self.assertEqual(instruction.destination, "VALIDATION_ROUTER")
            self.assertNotIn("protocol_diagnostics", feedback.to_dict())
            self.assertNotIn("input_digest", feedback.to_dict())

    def test_guard_accumulates_exact_seed_bundle_to_multi_seed_signal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(
                Path(raw) / "c-private-audit"
            )
            port = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            dispositions = []
            for seed, metric in (
                ("2026", 0.27),
                ("2027", 0.28),
                ("2028", 0.29),
            ):
                self.assertEqual(
                    port.pre_run(candidate(seed_id=seed)).status,
                    PortStatus.ALLOW,
                )
                dispositions.append(
                    port.post_run(
                        result(
                            raw_identity=f"raw-{seed}",
                            seed_id=seed,
                            ndcg=metric,
                        )
                    ).evidence_use
                )
            self.assertEqual(
                dispositions,
                [
                    "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
                    "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
                    (
                        "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_"
                        "DEVELOPMENT_SIGNAL"
                    ),
                ],
            )
            snapshot = ledger.evidence_snapshot()
            self.assertEqual(
                tuple(
                    item.observation_seed
                    for item in snapshot.observations
                ),
                ("2026", "2027", "2028"),
            )
            self.assertEqual(
                len(
                    snapshot.guard_core_projection(
                        snapshot_id=snapshot.digest,
                        claim_id="CLAIM-001",
                        protocol_id="PROTO-ML1M-FULL-001",
                    )["observation_ids"]
                ),
                3,
            )

    def test_guard_snapshot_excludes_unrelated_candidate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(
                Path(raw) / "c-private-audit"
            )
            unrelated = candidate("cand-unrelated")
            ledger.record_evidence_observation(
                candidate_semantic_digest=(
                    unrelated.candidate_semantic_digest
                ),
                protocol_digest=sha256_digest(protocol()),
                comparator_identity=unrelated.comparator,
                observation_seed="2026",
                observation_id=hashlib.sha256(
                    b"unrelated-observation"
                ).hexdigest(),
                raw_result=result(
                    "cand-unrelated",
                    raw_identity="raw-unrelated",
                ).to_dict(),
            )
            port = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            target = candidate()
            self.assertEqual(port.pre_run(target).status, PortStatus.ALLOW)
            stored = ledger.request_and_event_for_candidate(
                phase="PRE",
                candidate_id=target.candidate_id,
            )
            self.assertIsNotNone(stored)
            request, _event = stored
            self.assertEqual(
                request["context"]["current_evidence"]["observation_ids"],
                [],
            )

    def test_post_observed_protocol_flip_is_excluded_from_current_frontier(self) -> None:
        sampled = protocol()
        sampled["protocol_id"] = "PROTO-SAMPLED"
        sampled["evaluation_candidate_universe"] = {"mode": "sampled", "size": 100}
        with tempfile.TemporaryDirectory() as raw:
            port = EvidenceGuardPortV1(
                context(),
                EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit"),
                "opaque-c",
            )
            port.pre_run(candidate())
            feedback = DeterministicHelixFusionV1().fuse(
                port.post_run(result(observed=sampled))
            ).compact_feedback
            self.assertEqual(feedback.protocol_status, "PROTOCOL_BRANCH")
            self.assertEqual(
                HelixFusionBridgeV1().map(feedback).destination,
                "EXCLUDE_FROM_CURRENT_FRONTIER",
            )

    def test_pre_create_once_and_request_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit")
            port = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            item = candidate()
            self.assertEqual(port.pre_run(item), port.pre_run(item))
            self.assertEqual(ledger.count(), 1)
            changed = candidate(
                common_plan_digest=hashlib.sha256(b"different-plan").hexdigest()
            )
            with self.assertRaises(GuardLedgerError):
                port.pre_run(changed)
            self.assertEqual(ledger.count(), 1)

    def test_adapter_rejects_wrong_arm_candidate_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            port = EvidenceGuardPortV1(
                context(),
                EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit"),
                "opaque-c",
            )
            with self.assertRaises(ValueError):
                port.pre_run(candidate(arm_id="opaque-b"))
            port.pre_run(candidate())
            with self.assertRaises(ValueError):
                port.post_run(result(arm_id="opaque-b"))

    def test_post_can_recover_committed_pre_after_adapter_restart(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = EvidenceGuardLedgerWriterV1(Path(raw) / "c-private-audit")
            first = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            self.assertEqual(first.pre_run(candidate()).status, PortStatus.ALLOW)
            restarted = EvidenceGuardPortV1(context(), ledger, "opaque-c")
            post = restarted.post_run(result())
            self.assertEqual(post.status, PortStatus.ADJUDICATED)
            self.assertEqual(ledger.count(), 2)

    def test_reference_disposition_is_arm_blind_and_matches_four_cases(self) -> None:
        fixture = json.loads(
            (
                ROOT
                / "docs/research_line/m3/REFERENCE_DISPOSITION_FIXTURE_V1.json"
            ).read_text()
        )
        self.assertTrue(fixture["arm_blind"])
        self.assertNotIn("arm", json.dumps(fixture["cases"]).lower())
        self.assertEqual(len(fixture["cases"]), 4)
        flipped = protocol()
        flipped["protocol_id"] = "PROTO-SAMPLED"
        flipped["evaluation_candidate_universe"] = {"mode": "sampled", "size": 100}
        actual = {}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            exact = EvidenceGuardPortV1(
                context(),
                EvidenceGuardLedgerWriterV1(root / "exact"),
                "opaque-c",
            )
            actual["PRE-EXACT-CURRENT-PROTOCOL"] = exact.pre_run(candidate()).status.value
            actual["POST-EXACT-SINGLE-SEED"] = exact.post_run(result()).outcome_class

            pre_flip = EvidenceGuardPortV1(
                context(),
                EvidenceGuardLedgerWriterV1(root / "pre-flip"),
                "opaque-c",
            )
            actual["PRE-PLANNED-PROTOCOL-FLIP"] = pre_flip.pre_run(
                candidate("cand-pre-flip", planned=flipped)
            ).status.value

            post_flip = EvidenceGuardPortV1(
                context(),
                EvidenceGuardLedgerWriterV1(root / "post-flip"),
                "opaque-c",
            )
            post_flip.pre_run(candidate("cand-post-flip"))
            actual["POST-OBSERVED-PROTOCOL-FLIP"] = post_flip.post_run(
                result("cand-post-flip", observed=flipped, raw_identity="raw-flip")
            ).outcome_class
        expected = {
            item["case_id"]: item["expected_disposition"] for item in fixture["cases"]
        }
        self.assertEqual(actual, expected)

    def test_fusion_and_adapter_do_not_import_research_router_or_meta(self) -> None:
        for name in ("guard_adapter.py", "fusion.py"):
            source = (SRC / "recclaw_core/helix" / name).read_text()
            imports = " ".join(
                alias.name
                for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            ).lower()
            self.assertNotIn("research", imports)
            self.assertNotIn("router", imports)
            self.assertNotIn("meta", imports)


if __name__ == "__main__":
    unittest.main()
