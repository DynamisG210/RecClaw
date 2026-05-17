from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import propose_candidate as propose  # noqa: E402
import validate_candidate_proposal as validate  # noqa: E402


def sample_schema() -> dict[str, object]:
    return {
        "required_fields": [
            "proposal_type",
            "candidate_id",
            "parent_candidate_id",
            "base_model",
            "category",
            "hypothesis",
            "runnable_level",
            "runner_type",
            "consumes",
            "expected_effect",
            "risk",
            "decision_rule",
        ],
        "allowed_runnable_levels": ["parameter_only", "config_only", "spec_only", "code_required"],
        "allowed_base_models": ["BPR", "LightGCN"],
        "allowed_runner_types": ["config_only", "model", "posthoc"],
        "allowed_proposal_types": ["tuning", "algorithmic_variant", "research_spec"],
        "allowed_implementation_roots": [
            "recclaw_ext/models/",
            "recclaw_ext/posthoc/",
            "configs/candidates/",
            "configs/candidate_registry.yaml",
        ],
    }


def tuning_proposal(candidate_id: str, overrides: dict[str, object]) -> dict[str, object]:
    return {
        "proposal_type": "tuning",
        "candidate_id": candidate_id,
        "parent_candidate_id": "cand_lightgcn_small_embedding",
        "base_model": "LightGCN",
        "category": "Efficiency & Serving Constraints",
        "hypothesis": "Tune small embedding parameters.",
        "runnable_level": "parameter_only",
        "runner_type": "config_only",
        "consumes": list(overrides),
        "parameter_overrides": overrides,
        "expected_effect": {"primary_metric": "ndcg@10", "direction": "increase"},
        "risk": {"recbole_core_change_required": False},
        "decision_rule": {"keep_if": "improves"},
        "evaluation_plan": {
            "primary_metric": "ndcg@10",
            "validation_seeds": [2026, 2027, 2028],
            "aggregation": "report mean and std",
        },
    }


class CandidateProposalTests(unittest.TestCase):
    def test_generator_skips_used_parent_parameter_signature(self) -> None:
        registry = [
            {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "category": "Efficiency & Serving Constraints",
                "wired": True,
                "runner_type": "config_only",
                "status": "implemented",
                "priority": "high",
                "consumes": ["embedding_size", "n_layers"],
            }
        ]
        memory = [
            {
                "candidate_id": "cand_lightgcn_small_embedding",
                "params": {"embedding_size": 32, "n_layers": 1},
            }
        ]

        proposals = propose.generate_tuning_proposals(
            registry=registry,
            experiment_log="",
            memory=memory,
            count=1,
            stamp="20260508_000000",
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(set(proposals[0]["parameter_overrides"]), {"embedding_size", "n_layers"})
        self.assertNotEqual(proposals[0]["parameter_overrides"], {"embedding_size": 32, "n_layers": 1})

    def test_validator_rejects_duplicate_parameter_signature_in_file(self) -> None:
        registry = {
            "cand_lightgcn_small_embedding": {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["embedding_size", "n_layers"],
            }
        }
        seen_ids: set[str] = set()
        seen_signatures: set[str] = set()

        first = validate.validate_one(
            tuning_proposal("proposal_a", {"embedding_size": 32, "n_layers": 1}),
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=seen_ids,
            seen_param_signatures=seen_signatures,
            memory_param_signatures=set(),
        )
        second = validate.validate_one(
            tuning_proposal("proposal_b", {"n_layers": 1, "embedding_size": 32}),
            line_no=2,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=seen_ids,
            seen_param_signatures=seen_signatures,
            memory_param_signatures=set(),
        )

        self.assertEqual(first["status"], validate.ACCEPTED)
        self.assertEqual(second["status"], validate.REJECTED)
        self.assertTrue(any("duplicated" in error for error in second["errors"]))

    def test_validator_rejects_parameter_signature_from_memory(self) -> None:
        registry = {
            "cand_lightgcn_small_embedding": {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["embedding_size", "n_layers"],
            }
        }
        memory_signatures = {
            validate.parent_param_signature(
                "cand_lightgcn_small_embedding",
                {"embedding_size": 64, "n_layers": 2},
            )
        }

        result = validate.validate_one(
            tuning_proposal("proposal_c", {"n_layers": 2, "embedding_size": 64}),
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=memory_signatures,
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("already run" in error for error in result["errors"]))

    def test_memory_signature_loader_canonicalizes_explicit_trial_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "candidate_id": "proposal_a",
                        "parent_candidate_id": "cand_lightgcn_small_embedding",
                        "decision": "discard",
                        "parameter_signature": 'cand_lightgcn_small_embedding::{"max_norm":1.0}',
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            signatures = validate.load_memory_param_signatures(path)

        self.assertIn('cand_lightgcn_small_embedding::{"max_norm":1}', signatures)

    def test_signature_ignores_protocol_seed(self) -> None:
        plain = validate.parent_param_signature(
            "cand_lightgcn_small_embedding",
            {"embedding_size": 64, "n_layers": 2},
        )
        seeded = validate.parent_param_signature(
            "cand_lightgcn_small_embedding",
            {"embedding_size": 64, "n_layers": 2, "seed": 2027},
        )

        self.assertEqual(plain, seeded)

    def test_signature_canonicalizes_integral_floats(self) -> None:
        integer = validate.parent_param_signature(
            "cand_lightgcn_small_embedding",
            {"max_norm": 1},
        )
        floating = validate.parent_param_signature(
            "cand_lightgcn_small_embedding",
            {"max_norm": 1.0},
        )

        self.assertEqual(integer, floating)

    def test_validator_rejects_mismatched_declared_signature(self) -> None:
        registry = {
            "cand_lightgcn_small_embedding": {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["embedding_size", "n_layers"],
            }
        }
        proposal = tuning_proposal("proposal_d", {"embedding_size": 64, "n_layers": 2})
        proposal["parameter_signature"] = 'cand_lightgcn_small_embedding::{"embedding_size":32,"n_layers":1}'

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("does not match" in error for error in result["errors"]))

    def test_validator_accepts_canonical_equivalent_declared_signature(self) -> None:
        registry = {
            "cand_lightgcn_small_embedding": {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["max_norm"],
            }
        }
        proposal = tuning_proposal("proposal_e", {"max_norm": 1})
        proposal["parameter_signature"] = 'cand_lightgcn_small_embedding::{"max_norm":1.0}'

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
        )

        self.assertEqual(result["status"], validate.ACCEPTED)

    def test_generated_proposal_includes_multiseed_evaluation_plan(self) -> None:
        parent = {
            "candidate_id": "cand_bpr_margin_loss",
            "base_model": "BPR",
            "category": "Objective & Optimization",
            "runner_type": "model",
        }

        proposal = propose.build_parameter_proposal(parent, {"margin": 0.1}, 1, "20260508_000000")
        plan = proposal["evaluation_plan"]

        self.assertEqual(plan["primary_metric"], "ndcg@10")
        self.assertGreaterEqual(len(set(plan["validation_seeds"])), 3)
        self.assertIn("parameter_signature", proposal)


if __name__ == "__main__":
    unittest.main()
