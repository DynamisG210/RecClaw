from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import lint_recclaw_space as lint_space  # noqa: E402


def load_default_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    return (
        lint_space.load_yaml(ROOT / "configs" / "action_space.yaml"),
        lint_space.load_yaml(ROOT / "configs" / "candidate_proposal_schema.yaml"),
        lint_space.load_yaml(ROOT / "configs" / "candidate_registry.yaml"),
        lint_space.load_yaml(ROOT / "configs" / "search_policy.yaml"),
    )


class RecClawSpaceLintTests(unittest.TestCase):
    def test_current_contract_has_no_lint_errors(self) -> None:
        report = lint_space.run_lint(
            action_space_path=ROOT / "configs" / "action_space.yaml",
            schema_path=ROOT / "configs" / "candidate_proposal_schema.yaml",
            registry_path=ROOT / "configs" / "candidate_registry.yaml",
            search_policy_path=ROOT / "configs" / "search_policy.yaml",
            candidate_config_dir=ROOT / "configs" / "candidates",
        )
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(report["summary"]["warnings"], 0)

    def test_new_rs_specific_action_types_are_in_schema(self) -> None:
        action_space, schema, _registry, _search_policy = load_default_payloads()
        expected = {
            "negative_sampling",
            "pairwise_loss",
            "graph_propagation",
            "graph_augmentation",
            "layer_interaction",
            "auxiliary_loss",
            "rank_aware_loss",
        }
        self.assertTrue(expected.issubset(set(action_space["action_types"])))
        self.assertTrue(expected.issubset(set(schema["allowed_action_types"])))

    def test_planned_unwired_parameters_are_not_in_runnable_space(self) -> None:
        action_space, _schema, _registry, _search_policy = load_default_payloads()
        parameter_space = set(action_space["parameter_space"])
        blocked = {"aug_type", "aug_ratio", "ssl_weight", "contrastive_view", "diversity_lambda", "exposure_decay"}
        self.assertEqual(blocked & parameter_space, set())

    def test_undefined_parameter_action_type_is_reported(self) -> None:
        action_space, schema, registry, search_policy = load_default_payloads()
        action_space = copy.deepcopy(action_space)
        action_space["parameter_space"]["embedding_size"]["action_types"].append("representation")

        issues = lint_space.lint_payloads(
            action_space=action_space,
            schema=schema,
            registry=registry,
            search_policy=search_policy,
            candidate_configs={},
        )

        self.assertTrue(
            any(
                issue.check == "parameter_space.action_types"
                and issue.object_id == "embedding_size"
                and "representation" in issue.message
                for issue in issues
            )
        )

    def test_registry_consumes_outside_action_space_is_reported(self) -> None:
        action_space, schema, registry, search_policy = load_default_payloads()
        registry = copy.deepcopy(registry)
        registry["candidates"].append(
            {
                "candidate_id": "cand_bad_param",
                "base_model": "LightGCN",
                "runner_type": "model",
                "status": "implemented",
                "wired": True,
                "consumes": ["undeclared_knob"],
            }
        )

        issues = lint_space.lint_payloads(
            action_space=action_space,
            schema=schema,
            registry=registry,
            search_policy=search_policy,
            candidate_configs={},
        )

        self.assertTrue(
            any(
                issue.check == "candidate_registry.consumes"
                and issue.object_id == "cand_bad_param"
                and "undeclared_knob" in issue.message
                for issue in issues
            )
        )

    def test_candidate_declared_local_parameter_is_lintable(self) -> None:
        action_space, schema, registry, search_policy = load_default_payloads()
        registry = copy.deepcopy(registry)
        candidate_id = "cand_custom_declared_local_parameter"
        registry["candidates"].append(
            {
                "candidate_id": candidate_id,
                "base_model": "Custom",
                "runner_type": "model",
                "status": "implemented",
                "wired": True,
                "consumes": ["embedding_size", "architecture_gate_alpha"],
                "new_parameters": [
                    {
                        "name": "architecture_gate_alpha",
                        "type": "number",
                        "default": 0.2,
                        "search_space": [0.1, 0.2, 0.5],
                    }
                ],
            }
        )

        issues = lint_space.lint_payloads(
            action_space=action_space,
            schema=schema,
            registry=registry,
            search_policy=search_policy,
            candidate_configs={
                f"configs/candidates/{candidate_id}.yaml": {
                    "candidate_id": candidate_id,
                    "model": "CustomDeclaredLocalParameter",
                    "embedding_size": 64,
                    "architecture_gate_alpha": 0.2,
                }
            },
        )

        self.assertFalse(any(issue.object_id == candidate_id for issue in issues))

    def test_compatible_models_conflict_for_runnable_candidate_is_reported(self) -> None:
        action_space, schema, registry, search_policy = load_default_payloads()
        action_space = copy.deepcopy(action_space)
        registry = copy.deepcopy(registry)
        action_space["parameter_space"]["lambda_norm"]["compatible_models"] = ["BPR"]
        registry["candidates"].append(
            {
                "candidate_id": "cand_lightgcn_norm_conflict",
                "base_model": "LightGCN",
                "runner_type": "model",
                "status": "implemented",
                "wired": True,
                "consumes": ["lambda_norm"],
            }
        )

        issues = lint_space.lint_payloads(
            action_space=action_space,
            schema=schema,
            registry=registry,
            search_policy=search_policy,
            candidate_configs={},
        )

        self.assertTrue(
            any(
                issue.check == "candidate_registry.compatible_models"
                and issue.object_id == "cand_lightgcn_norm_conflict"
                and "LightGCN" in issue.message
                for issue in issues
            )
        )

    def test_lablog_runtime_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses LabLog path"):
            lint_space.reject_lablog_path(ROOT / "RecClaw_LabLog" / "2026.05.24" / "action_space.yaml")


if __name__ == "__main__":
    unittest.main()
