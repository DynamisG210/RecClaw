from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerCallV1,
    original_canary_prompt,
    research_canary_prompt,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    CANARY_ROUNDS_PER_ARM,
    CANARY_SEARCH_SEED,
    RealCanaryOrchestratorV1,
    RealCanaryProposalBrokerV1,
    canary_budget,
)


TEMPLATES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
SCHEMA = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "canary_proposal_response_v1.schema.json"
)


def proposal(label: str, *, intent: str = "DISCOVERY", index: int = 0):
    choices = (
        ("LATENT_FACTOR", "PAIRWISE_RANKING", "UNIFORM", "objective"),
        (
            "LIGHT_GRAPH_PROPAGATION",
            "CONTRASTIVE_AUXILIARY",
            "UNIFORM",
            "self_supervision",
        ),
        (
            "MESSAGE_TRANSFORM_GRAPH",
            "PAIRWISE_RANKING",
            "UNIFORM",
            "architecture",
        ),
        (
            "LATENT_FACTOR",
            "GEOMETRY_REGULARIZATION",
            "UNIFORM",
            "geometry",
        ),
    )
    choice_index = index % len(choices)
    backbone, objective, sampler, axis = choices[choice_index]
    utility = 0.55 + 0.1 * choice_index
    return {
        "backbone": backbone,
        "candidate_label": label,
        "expected_signal": f"bounded signal {label}",
        "failure_mode": f"bounded failure {label}",
        "hypothesis": f"mechanistic hypothesis {label}",
        "mechanism_axis": axis,
        "objective": objective,
        "proposal_intent": intent,
        "sampler": sampler,
        "utility_features": {
            "blocker_risk": 0.1,
            "cost": 0.3,
            "frontier_potential": utility,
            "information_gain": utility,
            "runnable_probability": 0.9,
            "useful_signal": utility,
        },
    }


class FakeUpstream:
    def __init__(self):
        self.calls = {}

    def call(self, *, logical_call_id, prompt, expected_proposal_count):
        if logical_call_id in self.calls:
            return self.calls[logical_call_id]
        if logical_call_id.startswith("original-"):
            proposals = [
                proposal(f"{logical_call_id}-{index}", index=index)
                for index in range(4)
            ]
        else:
            role = logical_call_id.rsplit("-", 1)[-1]
            # Role names contain underscores and remain the final suffix.
            role = next(
                item
                for item in (
                    "mechanism_composer",
                    "lineage_refiner",
                    "falsification_designer",
                    "frontier_architect",
                )
                if logical_call_id.endswith(item)
            )
            role_index = (
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            ).index(role)
            proposals = [
                proposal(
                    role,
                    intent=(
                        "FALSIFICATION"
                        if role == "falsification_designer"
                        else "DISCOVERY"
                    ),
                    index=role_index,
                )
            ]
        self.assert_prompt(prompt)
        self.assert_count(proposals, expected_proposal_count)
        response = {"proposals": proposals}
        call = CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest="1" * 64,
            response_digest=__import__("hashlib").sha256(
                json.dumps(response, sort_keys=True).encode()
            ).hexdigest(),
            response=response,
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=50,
            total_tokens=150,
            latency_ms=10,
            returned_model="gpt-5.4",
        )
        self.calls[logical_call_id] = call
        return call

    @staticmethod
    def assert_prompt(prompt):
        lowered = prompt.lower()
        for forbidden in (
            "bpr_mf",
            "directau",
            "lightgcn",
            "ngcf",
            "sgl",
            "ultragcn",
            "claim ceiling",
            "evidence admission",
        ):
            if forbidden in lowered:
                raise AssertionError(f"forbidden prompt token: {forbidden}")

    @staticmethod
    def assert_count(proposals, expected):
        if len(proposals) != expected:
            raise AssertionError("wrong proposal count")

    def call_count(self):
        return len(self.calls)


class M5RealCanaryTest(unittest.TestCase):
    def test_schema_and_prompts_are_closed_and_do_not_expose_anchor_recipes(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["proposals"]["maxItems"], 4)
        prompts = [
            original_canary_prompt(round_index=1, search_seed=CANARY_SEARCH_SEED)
        ] + [
            research_canary_prompt(
                role=role, round_index=1, search_seed=CANARY_SEARCH_SEED
            )
            for role in (
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            )
        ]
        for prompt in prompts:
            FakeUpstream.assert_prompt(prompt)

    def test_b_and_c_use_arm_private_calls_and_candidate_instances(self) -> None:
        upstream = FakeUpstream()
        broker = RealCanaryProposalBrokerV1.create(
            upstream=upstream,
            template_path=TEMPLATES,
        )
        b = broker.generate(
            arm=ArmCode.B,
            round_index=1,
            search_seed=CANARY_SEARCH_SEED,
            drafts=(),
            ceilings=canary_budget(),
        )
        c = broker.generate(
            arm=ArmCode.C,
            round_index=1,
            search_seed=CANARY_SEARCH_SEED,
            drafts=(),
            ceilings=canary_budget(),
        )
        self.assertEqual(upstream.call_count(), 8)
        self.assertEqual(b.proposal_session_digest, c.proposal_session_digest)
        self.assertEqual(b.ordered_programs, c.ordered_programs)
        self.assertEqual(b.input_tokens, c.input_tokens)
        self.assertEqual(b.output_tokens, c.output_tokens)
        self.assertEqual(b.billed_tokens, c.billed_tokens)
        sharing = broker.call_sharing_audit()
        self.assertEqual(sharing["policy"], "ARM_PRIVATE")
        self.assertEqual(sharing["physical_call_identities"], 8)
        self.assertEqual(sharing["consumer_logical_identities"], 8)
        self.assertEqual(sharing["cross_arm_physical_identities"], 0)
        self.assertEqual(broker.bc_controller_identity_digest, broker.bc_controller_identity_digest)
        first_primitives = {
            component["primitive_id"]
            for component in b.ordered_programs[0]["program_payload"]["components"]
            if "primitive_id" in component
        }
        self.assertIn("regularizer.alignment", first_primitives)

    def test_three_round_fake_upstream_canary_closes_records_and_active_meta(self) -> None:
        upstream = FakeUpstream()
        broker = RealCanaryProposalBrokerV1.create(
            upstream=upstream,
            template_path=TEMPLATES,
        )
        with tempfile.TemporaryDirectory() as raw, RealCanaryOrchestratorV1(
            Path(raw) / "canary", broker=broker
        ) as orchestrator:
            rounds = orchestrator.run_canary()
            self.assertEqual(len(rounds), CANARY_ROUNDS_PER_ARM)
            self.assertTrue(all(len(item) == 3 for item in rounds))
            self.assertTrue(
                all(
                    not result.training_backend_started
                    for triplet in rounds
                    for result in triplet
                )
            )
            audit = orchestrator.canary_audit()
            self.assertEqual(audit["round_count"], 9)
            self.assertEqual(audit["feedback_count"], 9)
            self.assertEqual(audit["execution_count"], 9)
            self.assertEqual(audit["guard_call_count"], 6)
            self.assertEqual(audit["broker_successful_upstream_calls"], 19)
            self.assertEqual(
                audit["barriers"], [[1, 7, 1], [2, 7, 1], [3, 7, 1]]
            )
            self.assertEqual(
                broker.research_controllers[ArmCode.B].policy.version, 4
            )
            self.assertEqual(
                broker.research_controllers[ArmCode.C].policy.version, 3
            )
            self.assertEqual(
                audit["state_store_integrity"]["integrity_check"], "ok"
            )
            guard_db = orchestrator.guard_ledger.db_path
            connection = __import__("sqlite3").connect(guard_db)
            post_events = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT full_event_json FROM guard_calls WHERE phase='POST'"
                )
            ]
            connection.close()
            self.assertEqual(len(post_events), 3)
            self.assertTrue(
                all(
                    item["evidence_admissibility"]["development_disposition"]
                    == "RECORD_EXECUTABILITY_ONLY"
                    for item in post_events
                )
            )


if __name__ == "__main__":
    unittest.main()
