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

    def test_validator_rejects_model_incompatible_action_space_parameter(self) -> None:
        action_space = validate.load_action_space(ROOT / "configs" / "action_space.yaml")
        registry = {
            "cand_bpr_bad_layers": {
                "candidate_id": "cand_bpr_bad_layers",
                "base_model": "BPR",
                "wired": True,
                "consumes": ["n_layers"],
            }
        }
        proposal = tuning_proposal("proposal_bad_layers", {"n_layers": 2})
        proposal["parent_candidate_id"] = "cand_bpr_bad_layers"
        proposal["base_model"] = "BPR"
        proposal["action_type"] = "parameter_tuning"

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space=action_space,
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("n_layers is not compatible" in error for error in result["errors"]))

    def test_validator_rejects_explicit_conditional_validity_violation(self) -> None:
        action_space = validate.load_action_space(ROOT / "configs" / "action_space.yaml")
        registry = {
            "cand_lightgcn_residual": {
                "candidate_id": "cand_lightgcn_residual",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["residual_weight", "n_layers"],
            }
        }
        proposal = tuning_proposal("proposal_bad_residual", {"residual_weight": 0.1, "n_layers": 1})
        proposal["parent_candidate_id"] = "cand_lightgcn_residual"
        proposal["action_type"] = "parameter_tuning"

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space=action_space,
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("residual_weight requires n_layers >=2" in error for error in result["errors"]))

    def test_validator_allows_single_axis_tuning_when_parent_context_is_implicit(self) -> None:
        action_space = validate.load_action_space(ROOT / "configs" / "action_space.yaml")
        registry = {
            "cand_lightgcn_residual": {
                "candidate_id": "cand_lightgcn_residual",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["residual_weight", "n_layers"],
            }
        }
        proposal = tuning_proposal("proposal_residual_only", {"residual_weight": 0.1})
        proposal["parent_candidate_id"] = "cand_lightgcn_residual"
        proposal["action_type"] = "parameter_tuning"

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space=action_space,
        )

        self.assertEqual(result["status"], validate.ACCEPTED)

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

    def test_validator_rejects_parameter_value_outside_action_space(self) -> None:
        registry = {
            "cand_lightgcn_small_embedding": {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "wired": True,
                "consumes": ["embedding_size", "n_layers"],
            }
        }
        proposal = tuning_proposal("proposal_out_of_space", {"embedding_size": 256, "n_layers": 2})

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space={
                "parameter_space": {
                    "embedding_size": {"values": [32, 64, 128]},
                    "n_layers": {"values": [1, 2, 3]},
                }
            },
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("outside action_space" in error for error in result["errors"]))

    def test_validator_rejects_consumes_outside_action_space(self) -> None:
        registry = {
            "cand_bpr_margin_loss": {
                "candidate_id": "cand_bpr_margin_loss",
                "base_model": "BPR",
                "wired": True,
                "consumes": ["margin"],
            }
        }
        proposal = {
            "proposal_type": "algorithmic_variant",
            "candidate_id": "proposal_new_param",
            "parent_candidate_id": "cand_bpr_margin_loss",
            "base_model": "BPR",
            "category": "Objective & Optimization",
            "action_type": "local_loss",
            "hypothesis": "Try a new local loss parameter.",
            "runnable_level": "code_required",
            "runner_type": "model",
            "consumes": ["margin", "surprise_alpha"],
            "new_parameters": [{"name": "surprise_alpha", "search_space": [0.1, 0.2]}],
            "implementation_plan": {
                "files": ["recclaw_ext/models/surprise.py"],
                "entrypoint": "recclaw_ext.models.surprise:SurpriseBPR",
            },
            "allowed_files": ["recclaw_ext/models/surprise.py"],
            "expected_effect": {"primary_metric": "ndcg@10", "direction": "increase"},
            "risk": {"recbole_core_change_required": False},
            "decision_rule": {"keep_if": "improves"},
        }

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema=sample_schema(),
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space={
                "action_types": {"local_loss": {}},
                "parameter_space": {"margin": {"values": [0.1, 0.2, 0.5]}},
                "allowed_implementation_roots": ["recclaw_ext/models/"],
            },
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("consumes includes parameters outside action_space" in error for error in result["errors"]))

    def test_validator_prefers_action_space_action_types(self) -> None:
        registry = {
            "cand_bpr_margin_loss": {
                "candidate_id": "cand_bpr_margin_loss",
                "base_model": "BPR",
                "wired": True,
                "consumes": ["margin"],
            }
        }
        proposal = tuning_proposal("proposal_action_type", {"margin": 0.1})
        proposal["parent_candidate_id"] = "cand_bpr_margin_loss"
        proposal["base_model"] = "BPR"
        proposal["runner_type"] = "model"
        proposal["action_type"] = "posthoc_rerank"

        result = validate.validate_one(
            proposal,
            line_no=1,
            schema={**sample_schema(), "allowed_action_types": ["posthoc_rerank"]},
            registry_by_id=registry,
            registry_ids=set(registry),
            seen_ids=set(),
            seen_param_signatures=set(),
            memory_param_signatures=set(),
            action_space={
                "action_types": {"parameter_tuning": {}},
                "parameter_space": {"margin": {"values": [0.1, 0.2, 0.5]}},
                "allowed_implementation_roots": ["recclaw_ext/models/"],
            },
        )

        self.assertEqual(result["status"], validate.REJECTED)
        self.assertTrue(any("action_type must be one of" in error for error in result["errors"]))

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
        self.assertEqual(proposal["action_type"], "parameter_tuning")

    def test_algorithmic_generator_skips_already_wired_template_stub(self) -> None:
        registry = [
            {
                "candidate_id": "cand_bpr_hard_negative_mix",
                "base_model": "BPR",
                "category": "Bias & Sample Construction",
                "status": "implemented",
                "wired": True,
                "runner_type": "model",
                "consumes": ["hard_negative_ratio"],
            },
            {
                "candidate_id": "cand_bpr_hard_negative_margin",
                "base_model": "BPR",
                "category": "Bias & Sample Construction",
                "status": "implemented",
                "wired": True,
                "runner_type": "model",
                "consumes": ["hard_negative_ratio", "margin"],
            },
        ]

        proposals = propose.generate_algorithmic_proposals(
            registry=registry,
            count=1,
            stamp="20260526_000000",
        )

        self.assertEqual(proposals, [])

    def test_algorithmic_generator_skips_spec_only_by_default(self) -> None:
        registry = [
            {
                "candidate_id": "cand_rerank_coverage_boost",
                "base_model": "LightGCN",
                "category": "Result Distribution Quality",
                "status": "implement-ready",
                "wired": False,
                "runner_type": "posthoc",
                "consumes": ["lambda_coverage"],
            }
        ]

        proposals = propose.generate_algorithmic_proposals(
            registry=registry,
            count=1,
            stamp="20260526_000000",
        )
        spec_proposals = propose.generate_algorithmic_proposals(
            registry=registry,
            count=1,
            stamp="20260526_000000",
            include_spec_only=True,
        )

        self.assertEqual(proposals, [])
        self.assertEqual(len(spec_proposals), 1)
        self.assertEqual(spec_proposals[0]["runnable_level"], "spec_only")


if __name__ == "__main__":
    unittest.main()
