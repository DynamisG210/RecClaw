from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1 import (  # noqa: E402
    ArmCode,
    DeterministicFusionV1,
    EvidenceAdjudicationStatus,
    MetaPolicyModeV1,
    NullEvidencePortV1,
    OriginalControllerV1,
    PhysicalInvocationPolicyV1,
    ProducerExecutionModeV1,
    ProposalGenerationSessionV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1 import controllers  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ExperimentContractV1,
    validate_equal_session_resource_ceilings,
    validate_no_research_evidence_authority_fields,
)


CONTRACT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "helix_abc_001"
    / "experiment_contract.v1.json"
)
GOLDEN_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "helix_abc_v1"
    / "original_controller_golden_v1.json"
)


class M0ContractTests(unittest.TestCase):
    def test_machine_contract_matches_the_single_python_truth(self) -> None:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        loaded = ExperimentContractV1.from_dict(payload)
        expected = default_experiment_contract()

        self.assertEqual(loaded.to_dict(), expected.to_dict())
        self.assertEqual(
            loaded.identity_digest,
            # Current delivered BL-ICF resource identity, shared by all arms.
            "3116ee1944402a49ae4c347341210ec212f91bd1deba79f3087c94ca514765d2",
        )

    def test_exact_arm_tuple_and_non_guard_equality(self) -> None:
        contract = default_experiment_contract()
        arm_a, arm_b, arm_c = contract.arm_policies

        self.assertEqual(tuple(item.arm for item in contract.arm_policies), tuple(ArmCode))
        self.assertFalse(arm_a.research_line_enabled)
        self.assertEqual(arm_a.evidence_port.value, "NullEvidencePortV1")
        self.assertEqual(arm_b.evidence_port.value, "NullEvidencePortV1")
        self.assertEqual(arm_c.evidence_port.value, "EvidenceGuardPortV1")
        self.assertEqual(arm_b.non_guard_projection(), arm_c.non_guard_projection())
        for field_name in (
            "bl_icf_search_space_digest",
            "common_execution_guard_digest",
            "deterministic_fusion_digest",
        ):
            self.assertEqual(
                len({getattr(item, field_name) for item in contract.arm_policies}), 1
            )

    def test_old_narrow_arm_a_and_forbidden_tuples_fail(self) -> None:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        payload["arm_policies"][0]["bl_icf_search_space_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact-equal"):
            ExperimentContractV1.from_dict(payload)

        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        payload["arm_policies"][1]["evidence_port"] = "EvidenceGuardPortV1"
        with self.assertRaisesRegex(ValueError, "wrong EvidencePort"):
            ExperimentContractV1.from_dict(payload)

    def test_experiment_contract_rejects_unknown_or_coerced_fields(self) -> None:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        payload["unexpected_authority_hint"] = "IGNORED"
        with self.assertRaisesRegex(ValueError, "closed and type-exact"):
            ExperimentContractV1.from_dict(payload)

        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        payload["scheduled_slots_per_arm_seed"] = "50"
        with self.assertRaisesRegex(ValueError, "closed and type-exact"):
            ExperimentContractV1.from_dict(payload)

    def test_closed_producer_and_meta_enums_have_no_runtime(self) -> None:
        self.assertEqual(
            {item.value for item in ProducerExecutionModeV1},
            {
                "BATCHED_ROLE_PORTFOLIO_V1",
                "NEUTRAL_MULTISAMPLE_CONTROL_V1",
                "BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1",
            },
        )
        self.assertEqual(
            {item.value for item in MetaPolicyModeV1},
            {"VERSIONED_POLICY_UPDATE", "STATIC_RESEARCH_ROUTER"},
        )
        self.assertNotIn("invoke", inspect.getsource(ProducerExecutionModeV1).lower())

    def test_session_equality_allows_different_future_call_shapes(self) -> None:
        ceilings = ResourceCeilingsV1(
            total_input_tokens=8000,
            total_output_tokens=4000,
            total_billed_token_debit=12000,
            total_proposal_count=8,
            wall_time_ms=300000,
            retry_debit=2,
            proposal_attempt_debit=8,
            ordinary_executions=1,
            common_validation_count=8,
            gpu_device_time_ms=3600000,
            gpu_cost_microunits=1000000,
        )
        mode = ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        sessions = (
            ProposalGenerationSessionV1(
                session_id="session-a",
                arm=ArmCode.A,
                round_id="round-a",
                resource_ceilings=ceilings,
                physical_invocation_policy=(
                    PhysicalInvocationPolicyV1.ONE_ORIGINAL_INVOCATION
                ),
                producer_execution_mode=None,
            ),
            ProposalGenerationSessionV1(
                session_id="session-b",
                arm=ArmCode.B,
                round_id="round-b",
                resource_ceilings=ceilings,
                physical_invocation_policy=(
                    PhysicalInvocationPolicyV1.BOUNDED_MODE_DEFINED_INVOCATIONS
                ),
                producer_execution_mode=mode,
            ),
            ProposalGenerationSessionV1(
                session_id="session-c",
                arm=ArmCode.C,
                round_id="round-c",
                resource_ceilings=ceilings,
                physical_invocation_policy=(
                    PhysicalInvocationPolicyV1.BOUNDED_MODE_DEFINED_INVOCATIONS
                ),
                producer_execution_mode=mode,
            ),
        )

        validate_equal_session_resource_ceilings(sessions)
        self.assertTrue(all(not item.llm_runtime_implemented for item in sessions))
        altered = replace(ceilings, total_output_tokens=3999)
        with self.assertRaisesRegex(ValueError, "exact-equal"):
            validate_equal_session_resource_ceilings(
                (sessions[0], sessions[1], replace(sessions[2], resource_ceilings=altered))
            )

    def test_null_port_never_returns_permission_or_admission(self) -> None:
        port = NullEvidencePortV1()
        pre = port.pre_run({"candidate_id": "candidate-1"})
        post = port.post_run({"candidate_id": "candidate-1"})

        self.assertIs(pre.status, EvidenceAdjudicationStatus.NOT_ADJUDICATED)
        self.assertIs(post.status, EvidenceAdjudicationStatus.NOT_ADJUDICATED)
        serialized = json.dumps([pre.to_dict(), post.to_dict()], sort_keys=True)
        self.assertNotIn("ALLOW", serialized)
        self.assertNotIn("ADMISSIBLE", serialized)
        self.assertNotIn("permission", serialized.lower())

    def test_deterministic_fusion_uses_only_null_status_and_projection(self) -> None:
        port = NullEvidencePortV1()
        adjudication = port.post_run({"candidate_id": "candidate-1"})
        fusion = DeterministicFusionV1()
        first = fusion.fuse(
            candidate_id="candidate-1",
            raw_outcome_projection={"outcome_class": "FIXTURE_RESULT"},
            adjudication=adjudication,
        )
        second = fusion.fuse(
            candidate_id="candidate-1",
            raw_outcome_projection={"outcome_class": "FIXTURE_RESULT"},
            adjudication=adjudication,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.search_feedback_class.value, "BASELINE_RESULT")

    def test_research_contract_forbidden_evidence_authority_fields(self) -> None:
        validate_no_research_evidence_authority_fields(
            {"candidate_id": "candidate-1", "information_gain": 0.5}
        )
        for name in (
            "claim_ceiling",
            "evidence_admission",
            "protocol_branch",
            "cross_protocol_contamination",
            "claimCeiling",
            "evidence-admission",
            "protocol branch",
            "guardReasonCodes",
            "permission_decision",
        ):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                validate_no_research_evidence_authority_fields({name: "forbidden"})

    def test_original_source_tuple_is_resolved_from_git(self) -> None:
        fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        source = fixture["resolved_source"]
        blob = subprocess.run(
            [
                "git",
                "rev-parse",
                f"{source['pre_research_line_commit']}:{source['path']}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "rev-parse", f"{source['first_research_line_commit']}^"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        absent = subprocess.run(
            [
                "git",
                "cat-file",
                "-e",
                (
                    f"{source['pre_research_line_commit']}:"
                    f"{source['research_line_path_absent_at_source']}"
                ),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(blob, source["agent_blob_sha1"])
        self.assertEqual(parent, source["pre_research_line_commit"])
        self.assertNotEqual(absent.returncode, 0)
        self.assertEqual(controllers.ORIGINAL_SOURCE_COMMIT, parent)

    def test_original_golden_trace_replays_byte_stably(self) -> None:
        fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        controller = OriginalControllerV1()

        self.assertEqual(
            controller.replay_golden_fixture(fixture),
            fixture["trace_expected"],
        )


if __name__ == "__main__":
    unittest.main()
