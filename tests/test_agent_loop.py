from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import agent  # noqa: E402
import build_candidate_search_tree as search_tree  # noqa: E402
import build_experience_summary as experience  # noqa: E402
import collect_result  # noqa: E402
import implement_candidate_proposal as implement  # noqa: E402
import validate_candidate_proposal as validate  # noqa: E402


class AgentLoopTests(unittest.TestCase):
    def test_loop_mode_policies_cover_requested_modes(self) -> None:
        self.assertEqual(set(agent.LOOP_MODE_POLICIES), {"tuning", "mixed", "explore", "auto"})
        self.assertEqual(agent.LOOP_MODE_POLICIES["tuning"]["proposal_mode"], "conservative")
        self.assertTrue(agent.LOOP_MODE_POLICIES["explore"]["auto_implement"])
        self.assertTrue(agent.LOOP_MODE_POLICIES["auto"]["llm_plan"])

    def test_parameter_signature_ignores_seed_only_overrides(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        first = rec_agent._parameter_signature(
            "cand_lightgcn_small_embedding",
            {"embedding_size": 128, "n_layers": 2, "seed": 2026},
        )
        second = rec_agent._parameter_signature(
            "cand_lightgcn_small_embedding",
            {"n_layers": 2, "embedding_size": 128, "seed": 2028},
        )
        self.assertEqual(first, second)

    def test_execution_signature_collapses_native_lightgcn_aliases(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        params = {"embedding_size": 128, "n_layers": 2}
        shallow = rec_agent._execution_signature(
            {
                "candidate_id": "cand_lightgcn_shallow_layers",
                "base_model": "LightGCN",
                "runner_type": "config_only",
                "entrypoint": "recbole.model.general_recommender.lightgcn:LightGCN",
            },
            params,
        )
        small = rec_agent._execution_signature(
            {
                "candidate_id": "cand_lightgcn_small_embedding",
                "base_model": "LightGCN",
                "runner_type": "config_only",
                "entrypoint": "recbole.model.general_recommender.lightgcn:LightGCN",
            },
            params,
        )
        self.assertEqual(shallow, small)

    def test_choose_params_avoids_used_execution_signature(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                seed=1,
                parameter_space={"embedding_size": [128, 64], "n_layers": [2, 1]},
            )
        )
        candidate = {
            "candidate_id": "cand_lightgcn_shallow_layers",
            "base_model": "LightGCN",
            "runner_type": "config_only",
            "entrypoint": "recbole.model.general_recommender.lightgcn:LightGCN",
            "consumes": ["embedding_size", "n_layers"],
        }
        used = {
            rec_agent._execution_signature(
                candidate,
                {"embedding_size": 128, "n_layers": 2},
            )
        }
        params = rec_agent._choose_execution_unique_params(candidate, used)
        self.assertNotEqual(
            rec_agent._execution_signature(candidate, params),
            rec_agent._execution_signature(candidate, {"embedding_size": 128, "n_layers": 2}),
        )

    def test_registry_consumed_numeric_parameters_are_supported(self) -> None:
        registry = agent.load_yaml(ROOT / "configs" / "candidate_registry.yaml")
        consumed = {
            str(key)
            for candidate in registry.get("candidates", [])
            for key in (candidate.get("consumes") or [])
        }
        ignored = {"train_neg_sample_args"}
        missing_from_space = consumed - set(agent.AgentConfig().parameter_space) - ignored
        missing_from_schema = consumed - set(agent.PROPOSAL_PARAMETER_KEYS) - ignored
        self.assertEqual(missing_from_space, set())
        self.assertEqual(missing_from_schema, set())

    def test_agent_parameter_space_loads_from_action_space(self) -> None:
        self.assertIn("lambda_norm", agent.DEFAULT_PARAMETER_SPACE)
        self.assertEqual(agent.DEFAULT_PARAMETER_SPACE["hard_negative_ratio"], [0.25, 0.5, 0.75])

    def test_search_policy_loads_runtime_budget(self) -> None:
        policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        registry = agent.load_yaml(ROOT / "configs" / "candidate_registry.yaml")
        registry_ids = {str(item.get("candidate_id")) for item in registry.get("candidates", [])}
        priority_families = set(policy["search_stages"]["exploit"]["priority_families"])
        self.assertEqual(policy["protocol_lock"]["primary_metric"], "ndcg@10")
        self.assertEqual(policy["family_budget"]["max_code_required_per_window"], 1)
        self.assertEqual(priority_families - registry_ids, set())

    def test_wired_local_model_entrypoints_are_concrete_modules(self) -> None:
        registry = agent.load_yaml(ROOT / "configs" / "candidate_registry.yaml")
        package_level = []
        for candidate in registry.get("candidates", []):
            entrypoint = str(candidate.get("entrypoint") or "")
            if not bool(candidate.get("wired")) or not entrypoint.startswith("recclaw_ext."):
                continue
            module = entrypoint.split(":", 1)[0]
            if module in {"recclaw_ext.models", "recclaw_ext.posthoc"}:
                package_level.append(candidate.get("candidate_id"))
        self.assertEqual(package_level, [])

    def test_loose_json_parser_handles_nested_arrays(self) -> None:
        parsed = agent.RecClawAgent._parse_json_loose('prefix [{"a": [1, 2], "b": {"c": 3}}] suffix')
        self.assertEqual(parsed, [{"a": [1, 2], "b": {"c": 3}}])

    def test_auto_planner_respects_disabled_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    loop_mode="auto",
                    allow_llm_fallback=False,
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                )
            )

            def fail_chat(*_args: object, **_kwargs: object) -> str:
                raise RuntimeError("api down")

            rec_agent._chat_completion = fail_chat  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                rec_agent.apply_auto_planner(round_id=1)

    def test_planner_payload_sanitizes_extra_fields(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        payload, ignored = rec_agent._sanitize_planner_payload(
            {"action": "propose_mixed", "reason": "ok", "proposal_count": 4, "proposals": [{"x": 1}]}
        )
        self.assertEqual(payload, {"action": "propose_mixed", "reason": "ok", "proposal_count": 4})
        self.assertEqual(ignored, ["proposals"])

    def test_auto_round_policy_resets_mutated_explore_state(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig(loop_mode="auto"))
        rec_agent.config.proposal_mode = "explore"
        rec_agent.config.enable_candidate_proposals = False
        rec_agent.config.auto_implement_code_required = False
        rec_agent.reset_round_policy()
        self.assertEqual(rec_agent.config.proposal_mode, "mixed")
        self.assertTrue(rec_agent.config.enable_candidate_proposals)
        self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_experiment_directive_resolves_file_and_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directive_path = Path(tmp) / "directive.txt"
            directive_path.write_text("Focus LightGCNResidualNorm.", encoding="utf-8")
            resolved = agent.resolve_experiment_directive(
                "Avoid complex debias proposals.",
                directive_path,
            )
            self.assertIn("Focus LightGCNResidualNorm", resolved)
            self.assertIn("Avoid complex debias", resolved)
            self.assertEqual(
                agent.resolve_experiment_directive("Use this", directive_path, disabled=True),
                "",
            )

    def test_experience_builder_rejects_lablog_runtime_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses LabLog path"):
            experience.reject_lablog_path("RecClaw_LabLog/2026.05.17/summary.md")

    def test_experience_builder_outputs_insufficient_evidence(self) -> None:
        summary = experience.summarize_memory(
            [],
            action_space={},
            policy={"metric": "ndcg@10", "minimum_successful_runs": 3},
        )
        text = experience.build_markdown(summary, {"minimum_successful_runs": 3})
        self.assertIn("insufficient evidence", text)

    def test_experience_builder_tracks_crash_only_family(self) -> None:
        summary = experience.summarize_memory(
            [
                {
                    "candidate_id": "cand_lightgcn_residual_norm_trial",
                    "base_model": "LightGCN",
                    "status": "crash",
                    "decision": "revise",
                    "params": {"lambda_norm": 0.001, "max_norm": 1.0},
                }
            ],
            action_space={},
            policy={"metric": "ndcg@10", "minimum_successful_runs": 1},
        )
        family = summary["family_summaries"][0]
        self.assertEqual(family["policy_bucket"], "avoid")
        self.assertEqual(family["crash_count"], 1)
        self.assertIsNone(family["best"])
        self.assertEqual(family["action_type"], "regularization")

    def test_experience_builder_outputs_planner_policy(self) -> None:
        search_policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        summary = experience.summarize_memory(
            [
                {
                    "candidate_id": "cand_lightgcn_residual_norm_trial_a",
                    "parent_candidate_id": "cand_lightgcn_residual_norm",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "keep",
                    "params": {"lambda_norm": 0.001, "max_norm": 1.0},
                    "result": {"ndcg@10": 0.292},
                    "parameter_signature": 'cand_lightgcn_residual_norm::{"lambda_norm":0.001,"max_norm":1.0}',
                    "seed_validation": {"status": "passed", "mean": 0.291, "std": 0.001},
                },
                {
                    "candidate_id": "cand_lightgcn_residual_norm_trial_b",
                    "parent_candidate_id": "cand_lightgcn_residual_norm",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "keep",
                    "params": {"lambda_norm": 0.001, "max_norm": 0.5},
                    "result": {"ndcg@10": 0.289},
                    "parameter_signature": 'cand_lightgcn_residual_norm::{"lambda_norm":0.001,"max_norm":0.5}',
                },
                {
                    "candidate_id": "cand_lightgcn_debias_pareto_contrastive",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "discard",
                    "params": {"debias_alpha": 0.5, "pareto_temperature": 1.0, "cl_temperature": 0.5},
                    "result": {"ndcg@10": 0.19},
                },
            ],
            action_space={},
            policy=experience.load_yaml(ROOT / "configs" / "reflection_policy.yaml"),
            search_policy=search_policy,
        )
        policy = experience.build_experience_policy(summary, search_policy=search_policy, tree_summary={})
        self.assertIn("cand_lightgcn_residual_norm", policy["encourage_families"])
        self.assertTrue(policy["promising_parameter_regions"])
        self.assertTrue(policy["failed_compositions"])
        self.assertIn("do_not_repeat_signatures", policy)
        text = experience.build_markdown({**summary, "experience_policy": policy}, {"minimum_successful_runs": 1}, policy)
        self.assertIn("Prefer families", text)

    def test_experience_builder_dedupes_results_csv_by_run_id(self) -> None:
        memory_rows = [
            {
                "candidate_id": "cand_x",
                "run_id": "run_same",
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.3},
                "params": {"a": 1},
            }
        ]
        results_rows = [{"candidate_id": "cand_x", "run_id": "run_same", "status": "success", "ndcg@10": "0.3"}]
        combined = experience.combine_evidence_rows(memory_rows, results_rows)
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["params"], {"a": 1})

    def test_experience_builder_enriches_memory_metric_from_results_csv(self) -> None:
        memory_rows = [{"candidate_id": "cand_x", "run_id": "run_same", "result": {}, "params": {"a": 1}}]
        results_rows = [{"candidate_id": "cand_x", "run_id": "run_same", "status": "success", "ndcg@10": "0.31"}]
        combined = experience.combine_evidence_rows(memory_rows, results_rows)
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["result"]["ndcg@10"], "0.31")

    def test_experience_builder_matches_pairwise_failed_compositions(self) -> None:
        search_policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        summary = experience.summarize_memory(
            [
                {
                    "candidate_id": "cand_debias_pareto",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "discard",
                    "params": {"debias_alpha": 0.5, "pareto_temperature": 1.0},
                    "result": {"ndcg@10": 0.19},
                }
            ],
            action_space={},
            policy=experience.load_yaml(ROOT / "configs" / "reflection_policy.yaml"),
            search_policy=search_policy,
        )
        self.assertIn({"composition": "debias + pareto", "failures": 1}, summary["failed_compositions"])

    def test_results_csv_rows_can_feed_experience_summary(self) -> None:
        rows = [
            {
                "run_id": "candidate_cand_lightgcn_residual_layer_mix_20260518_010101",
                "model": "LightGCN",
                "status": "success",
                "ndcg@10": "0.281",
            }
        ]
        converted = experience.result_rows_to_memory_rows(rows)
        self.assertEqual(converted[0]["candidate_id"], "cand_lightgcn_residual_layer_mix")
        summary = experience.summarize_memory(
            converted,
            action_space={},
            policy={"metric": "ndcg@10", "minimum_successful_runs": 1},
        )
        self.assertEqual(summary["successful_metric_rows"], 1)

    def test_candidate_search_tree_builds_parent_child_metric_summary(self) -> None:
        tree = search_tree.build_tree(
            registry={
                "candidates": [
                    {
                        "candidate_id": "cand_parent",
                        "base_model": "LightGCN",
                        "status": "implemented",
                        "consumes": ["residual_weight"],
                    }
                ]
            },
            proposals=[
                {
                    "candidate_id": "cand_child",
                    "parent_candidate_id": "cand_parent",
                    "base_model": "LightGCN",
                    "action_type": "parameter_tuning",
                    "parameter_overrides": {"residual_weight": 0.1},
                }
            ],
            memory_rows=[
                {
                    "candidate_id": "cand_child",
                    "parent_candidate_id": "cand_parent",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "keep",
                    "result": {"ndcg@10": 0.3},
                    "seed_validation": {"status": "passed", "mean": 0.299},
                }
            ],
            results_rows=[],
            metric="ndcg@10",
        )
        by_id = {node["candidate_id"]: node for node in tree["nodes"]}
        self.assertIn("cand_child", by_id["cand_parent"]["children"])
        self.assertEqual(by_id["cand_child"]["best_ndcg@10"], 0.3)
        self.assertEqual(by_id["cand_child"]["decision_counts"]["keep"], 1)
        self.assertIn("cand_parent", search_tree.build_markdown(tree))
        self.assertIn("cand_child", search_tree.build_mermaid(tree))

    def test_candidate_search_tree_handles_missing_parent(self) -> None:
        tree = search_tree.build_tree(
            registry={"candidates": []},
            proposals=[{"candidate_id": "cand_child", "parent_candidate_id": "cand_missing_parent"}],
            memory_rows=[],
            results_rows=[],
            metric="ndcg@10",
        )
        by_id = {node["candidate_id"]: node for node in tree["nodes"]}
        self.assertIn("cand_missing_parent", by_id)
        self.assertIn("cand_child", by_id["cand_missing_parent"]["children"])

    def test_candidate_search_tree_non_ndcg_metric_does_not_fill_ndcg_alias(self) -> None:
        tree = search_tree.build_tree(
            registry={"candidates": [{"candidate_id": "cand_parent"}]},
            proposals=[{"candidate_id": "cand_child", "parent_candidate_id": "cand_parent"}],
            memory_rows=[{"candidate_id": "cand_child", "result": {"recall@10": 0.8}}],
            results_rows=[],
            metric="recall@10",
        )
        by_id = {node["candidate_id"]: node for node in tree["nodes"]}
        self.assertEqual(by_id["cand_child"]["best_recall_at_10"], 0.8)
        self.assertIsNone(by_id["cand_child"]["best_ndcg@10"])

    def test_candidate_search_tree_dedupes_results_csv_by_memory_run_id(self) -> None:
        tree = search_tree.build_tree(
            registry={"candidates": [{"candidate_id": "cand_x"}]},
            proposals=[],
            memory_rows=[
                {
                    "candidate_id": "cand_x",
                    "run_id": "run_same",
                    "status": "success",
                    "result": {"ndcg@10": 0.3},
                }
            ],
            results_rows=[
                {"candidate_id": "cand_x", "run_id": "run_same", "status": "success", "ndcg@10": "0.3"}
            ],
            metric="ndcg@10",
        )
        by_id = {node["candidate_id"]: node for node in tree["nodes"]}
        self.assertEqual(by_id["cand_x"]["run_count"], 1)

    def test_experience_tree_hints_use_child_metrics(self) -> None:
        hints = experience.tree_policy_hints(
            {
                "metric": "ndcg@10",
                "nodes": [
                    {"candidate_id": "parent", "children": ["child"], "best_ndcg@10": None},
                    {"candidate_id": "child", "children": [], "best_ndcg@10": 0.31},
                ],
            }
        )
        self.assertEqual(hints["high_yield_parents"][0]["family"], "parent")
        self.assertEqual(hints["high_yield_parents"][0]["best"], 0.31)

    def test_experiment_directive_enters_llm_context_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "agent_state_summary.json"
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                    state_summary_path=summary_path,
                    use_experiment_directive=True,
                    experiment_directive="Focus on validating LightGCNResidualNorm; avoid broad new code_required ideas.",
                )
            )
            rec_agent.observe()
            context = rec_agent._llm_proposal_context()
            self.assertTrue(context["experiment_directive"]["enabled"])
            self.assertIn("LightGCNResidualNorm", context["experiment_directive"]["text"])
            self.assertIn("experiment_directive", context["steering_priority"][1])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["experiment_directive"]["enabled"])
            self.assertIn("avoid broad", summary["experiment_directive"]["text"])

    def test_reflect_revises_baseline_win_without_history_best(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig(min_keep_delta=1e-5))
        decision, reason, _ = rec_agent.reflect(
            {"candidate_id": "cand_test"},
            {},
            {"exit_code": 0},
            {"status": "success"},
            {"delta": 0.003},
            {"delta": 0.0},
        )
        self.assertEqual(decision, "revise")
        self.assertIn("does not beat history best", reason)

    def test_auto_implementation_write_allowlist(self) -> None:
        self.assertTrue(implement.path_is_allowed("recclaw_ext/models/new_model.py"))
        self.assertFalse(implement.path_is_allowed("configs/candidates/cand_new.yaml"))
        self.assertFalse(implement.path_is_allowed("recclaw_ext/models/__init__.py"))
        self.assertFalse(implement.path_is_allowed("recclaw_ext/posthoc/__init__.py"))
        self.assertFalse(implement.path_is_allowed("recclaw_ext/models/../../configs/task_ml1m.yaml"))
        self.assertFalse(implement.path_is_allowed("/tmp/new_model.py"))
        self.assertFalse(implement.path_is_allowed("configs/task_ml1m.yaml"))
        self.assertFalse(implement.path_is_allowed("configs/candidate_registry.yaml"))

    def test_implementation_context_receives_experiment_directive(self) -> None:
        context = implement.build_implementation_context(
            {"candidate_id": "cand_x", "base_model": "LightGCN"},
            {"candidates": []},
            "Keep implementation minimal and avoid broad new mechanisms.",
        )
        self.assertTrue(context["experiment_directive"]["enabled"])
        self.assertIn("minimal", context["experiment_directive"]["text"])

    def test_proposal_validator_rejects_unsafe_implementation_paths(self) -> None:
        allowed = {"recclaw_ext/models/", "recclaw_ext/posthoc/"}
        self.assertTrue(validate.path_is_allowed("recclaw_ext/models/new_model.py", allowed))
        self.assertTrue(validate.path_is_allowed("recclaw_ext/models/", allowed))
        self.assertFalse(validate.path_is_allowed("recclaw_ext/models/__init__.py", allowed))
        self.assertFalse(validate.path_is_allowed("recclaw_ext/models/../../configs/task_ml1m.yaml", allowed))
        self.assertFalse(validate.path_is_allowed("configs/candidates/cand_new.yaml", allowed))

    def test_auto_implementation_refuses_existing_file_overwrite(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to overwrite existing implementation file"):
            implement.validate_files(
                [{"path": "recclaw_ext/models/bpr_margin.py", "content": "# accidental overwrite"}]
            )

    def test_auto_implementation_rejects_config_get_in_generated_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "config.get"):
            implement.validate_files(
                [
                    {
                        "path": "recclaw_ext/models/generated_config_get.py",
                        "content": "class X:\n    def __init__(self, config):\n        self.x = config.get('x', 1)\n",
                    }
                ]
            )

    def test_auto_implementation_rejects_undefined_self_reg_loss(self) -> None:
        with self.assertRaisesRegex(ValueError, "self.reg_loss"):
            implement.validate_static_model_code(
                "recclaw_ext/models/generated.py",
                "class X:\n"
                "    def calculate_loss(self, user_e, pos_e, neg_e):\n"
                "        return self.reg_loss(user_e, pos_e, neg_e)\n",
            )

    def test_auto_implementation_allows_lightgcn_inherited_reg_loss(self) -> None:
        implement.validate_static_model_code(
            "recclaw_ext/models/generated_lightgcn.py",
            "from recbole.model.general_recommender.lightgcn import LightGCN\n"
            "class X(LightGCN):\n"
            "    def calculate_loss(self, interaction):\n"
            "        return self.reg_loss(self.user_embedding.weight)\n",
        )

    def test_auto_implementation_allows_local_lightgcn_parent_reg_loss(self) -> None:
        implement.validate_static_model_code(
            "recclaw_ext/models/generated_lightgcn_child.py",
            "from recclaw_ext.models.lightgcn_residual import LightGCNResidualMix\n"
            "class X(LightGCNResidualMix):\n"
            "    def calculate_loss(self, interaction):\n"
                "        return self.reg_loss(self.user_embedding.weight)\n",
        )

    def test_auto_implementation_rejects_positional_soft_l2_scalar(self) -> None:
        with self.assertRaisesRegex(ValueError, "soft_l2_norm_penalty"):
            implement.validate_static_model_code(
                "recclaw_ext/models/generated_bad_norm.py",
                "from recclaw_ext.models._utils import soft_l2_norm_penalty\n"
                "class X:\n"
                "    def _norm_penalty(self, emb):\n"
                "        return soft_l2_norm_penalty(emb, self.max_norm)\n",
            )

    def test_auto_planner_gates_seed_verify_without_pending_keep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    loop_mode="auto",
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                )
            )
            parsed, reason = rec_agent._maybe_override_auto_action(
                {"action": "multi_seed_verify", "reason": "check stability", "proposal_count": 3}
            )
        self.assertEqual(parsed["action"], "propose_mixed")
        self.assertEqual(reason, "no_unverified_keep_for_seed_verify")

    def test_state_summary_tracks_validated_best_and_duplicates(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.memory = [
            {
                "round_id": 1,
                "candidate_id": "cand_a",
                "parent_candidate_id": "parent_a",
                "params": {"x": 1},
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.3},
                "seed_validation": {"status": "passed", "mean": 0.29},
                "parameter_signature": 'parent_a::{"x":1}',
            },
            {
                "event": "proposal_rejected",
                "proposal_id": "dup",
                "parameter_signature": 'parent_a::{"x":1}',
                "errors": ["parameter signature was already run in agent memory: parent_a::{\"x\":1}"],
            },
        ]
        rec_agent.best_by_model = {
            "LightGCN": {
                "run_id": "best_run",
                "model": "LightGCNResidualNorm",
                "status": "success",
                "ndcg@10": 0.31,
            }
        }
        summary = rec_agent._build_agent_state_summary()
        self.assertEqual(summary["validated_best"]["candidate_id"], "cand_a")
        self.assertIn('parent_a::{"x":1}', summary["recent_duplicate_rejections"])
        self.assertEqual(summary["best_by_model"]["LightGCN"]["run_id"], "best_run")

    def test_auto_implementation_rewrites_package_entrypoint(self) -> None:
        entrypoint = implement.normalize_entrypoint(
            "recclaw_ext.models:BPRHardNegative",
            [{"path": "recclaw_ext/models/bpr_hard_negative.py", "content": ""}],
        )
        self.assertEqual(entrypoint, "recclaw_ext.models.bpr_hard_negative:BPRHardNegative")

    def test_auto_implementation_rewrites_entrypoint_by_class_definition(self) -> None:
        entrypoint = implement.normalize_entrypoint(
            "recclaw_ext.models:CustomBPRVariant",
            [
                {"path": "recclaw_ext/models/first_file.py", "content": "class OtherModel:\n    pass\n"},
                {"path": "recclaw_ext/models/generated_variant.py", "content": "class CustomBPRVariant:\n    pass\n"},
            ],
        )
        self.assertEqual(entrypoint, "recclaw_ext.models.generated_variant:CustomBPRVariant")

    def test_auto_implementation_locks_proposal_parameter_defaults(self) -> None:
        config = {"candidate_id": "cand_x", "model": "X", "debias_alpha": 0.5}
        proposal = {"parameter_overrides": {"debias_alpha": 0.1, "embedding_size": 128}}
        locked = implement.enforce_proposal_parameter_defaults(config, proposal)
        self.assertEqual(locked["debias_alpha"], 0.1)
        self.assertEqual(locked["embedding_size"], 128)

    def test_compact_smoke_summary_detects_metric_stagnation(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["run_candidate"],
            returncode=0,
            stdout=(
                "epoch 0 evaluating [time: 1s, valid_score: 0.107700]\n"
                "epoch 1 evaluating [time: 1s, valid_score: 0.107700]\n"
                "epoch 2 evaluating [time: 1s, valid_score: 0.107700]\n"
                '{"run_id": "r1", "log_path": "/tmp/r1.log"}\n'
            ),
            stderr="",
        )
        summary = implement.compact_smoke_summary(completed, ["epochs=3", "stopping_step=2"])
        self.assertTrue(summary["metric_stagnation"])
        self.assertEqual(summary["valid_score_unique_count"], 1)
        self.assertEqual(summary["run_id"], "r1")

    def test_auto_implementation_rejects_external_entrypoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "must stay inside recclaw_ext"):
            implement.normalize_entrypoint(
                "recbole.model.general_recommender.bpr:BPR",
                [{"path": "recclaw_ext/models/new_model.py", "content": "class NewModel:\n    pass\n"}],
            )

    def test_auto_implementation_rejects_entrypoint_not_generated(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of the generated files"):
            implement.normalize_entrypoint(
                "recclaw_ext.models.bpr_margin:BPRMargin",
                [{"path": "recclaw_ext/models/new_model.py", "content": "class NewModel:\n    pass\n"}],
            )

    def test_collect_result_repairs_legacy_csv_header(self) -> None:
        legacy_header = [field for field in collect_result.CSV_FIELDS if field != "latency_ms"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(legacy_header)
                writer.writerow([""] * len(legacy_header))
                writer.writerow([""] * len(collect_result.CSV_FIELDS))
            collect_result.ensure_csv(path)
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], collect_result.CSV_FIELDS)
        self.assertEqual(len(rows[1]), len(collect_result.CSV_FIELDS))
        self.assertEqual(len(rows[2]), len(collect_result.CSV_FIELDS))

    def test_collect_result_does_not_use_runtime_as_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                '{"model": "LightGCN", "dataset": "ml-1m", "status": "success", '
                '"run_time": 12.3, "test_result": {"ndcg@10": 0.27}}\n',
                encoding="utf-8",
            )
            parsed = collect_result.parse_recbole_log(log_path)
        self.assertEqual(parsed["latency_ms"], "")
        self.assertIn("was not used as inference latency", "; ".join(parsed["warnings"]))

    def test_implemented_bpr_variants_are_exported(self) -> None:
        init_text = (ROOT / "recclaw_ext" / "models" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("BPRLongTailReweight", init_text)
        self.assertIn("BPRPopularityRegularized", init_text)
        self.assertIn("BPRNormConstrained", init_text)

    def test_auto_implementation_skips_promoted_known_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(memory_path=Path(tmp) / "agent_memory.jsonl")
            )
            rec_agent.registry = [
                {
                    "candidate_id": "cand_bpr_long_tail_reweight",
                    "status": "implemented",
                    "wired": True,
                }
            ]
            rec_agent.candidate_proposals = [
                {
                    "candidate_id": "cand_bpr_long_tail_reweight_impl_001",
                    "parent_candidate_id": "cand_bpr_long_tail_reweight",
                }
            ]
            rec_agent.proposal_validation_report = {
                "results": [
                    {
                        "candidate_id": "cand_bpr_long_tail_reweight_impl_001",
                        "status": "needs_review",
                        "runnable_level": "code_required",
                    }
                ]
            }

            def fail_command(*_args: object, **_kwargs: object) -> dict[str, object]:
                raise AssertionError("implementation command should not run")

            rec_agent._run_json_command = fail_command  # type: ignore[method-assign]
            rec_agent.implement_needs_review_proposals(round_id=1)
            self.assertEqual(rec_agent.memory[-1]["event"], "implementation_skipped")

    def test_terminal_implementation_skip_clears_pending_review_gate(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.memory = [
            {
                "event": "proposal_needs_review",
                "proposal_id": "cand_auto_parent_impl_001",
                "next_action": "promote_to_implementation_queue",
            },
            {
                "event": "implementation_skipped",
                "proposal_id": "cand_auto_parent_impl_001",
                "reason": "known parent was auto-promoted and is runnable",
            },
        ]
        self.assertFalse(rec_agent._has_pending_code_required_review())
        payload, reason = rec_agent._maybe_override_auto_action(
            {"action": "implement_needs_review", "reason": "try implementation"}
        )
        self.assertEqual(payload["action"], "propose_mixed")
        self.assertEqual(reason, "no_actionable_code_review_pending")

    def test_llm_proposal_repair_fills_new_parameters_and_unique_id(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_parent",
                "consumes": ["embedding_size"],
            },
            {
                "candidate_id": "cand_existing",
            },
        ]
        proposal = {
            "candidate_id": "cand_existing",
            "parent_candidate_id": "cand_parent",
            "runnable_level": "code_required",
            "consumes": ["embedding_size", "lambda_new"],
            "new_parameters": [],
        }
        rec_agent._repair_llm_proposal_metadata(
            proposal,
            {"cand_parent": rec_agent.registry[0], "cand_existing": rec_agent.registry[1]},
            {"cand_existing"},
        )
        self.assertNotEqual(proposal["candidate_id"], "cand_existing")
        self.assertIn("lambda_new", proposal["new_parameters"])

    def test_soft_l2_norm_penalty_accepts_legacy_positional_max_norm(self) -> None:
        try:
            import importlib.util
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local test environment")

        spec = importlib.util.spec_from_file_location(
            "recclaw_test_utils",
            ROOT / "recclaw_ext" / "models" / "_utils.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        emb = torch.tensor([[3.0, 4.0]])
        legacy = module.soft_l2_norm_penalty(emb, 2.0)
        keyword = module.soft_l2_norm_penalty(emb, max_norm=2.0)
        self.assertAlmostEqual(float(legacy), float(keyword))

    def test_health_scan_quarantines_unsafe_local_model_source(self) -> None:
        temp_source = ROOT / "recclaw_ext" / "models" / "_tmp_bad_health.py"
        temp_source.write_text(
            "class Bad:\n"
            "    def __init__(self, config):\n"
            "        self.x = config.get('x', 1)\n",
            encoding="utf-8",
        )
        try:
            rec_agent = agent.RecClawAgent(agent.AgentConfig())
            rec_agent.registry = [
                {
                    "candidate_id": "cand_bad",
                    "wired": True,
                    "runner_type": "model",
                    "entrypoint": "recclaw_ext.models._tmp_bad_health:Bad",
                }
            ]
            issues = rec_agent._build_candidate_health_issues()
            self.assertIn("cand_bad", issues)
            self.assertIn("config.get", "; ".join(issues["cand_bad"]))
        finally:
            temp_source.unlink(missing_ok=True)

    def test_health_scan_quarantines_missing_candidate_config(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_missing_config_for_health",
                "wired": True,
                "runner_type": "config_only",
                "entrypoint": "recbole.model.general_recommender.bpr:BPR",
            }
        ]
        issues = rec_agent._build_candidate_health_issues()
        self.assertIn("cand_missing_config_for_health", issues)
        self.assertIn("candidate config is missing", "; ".join(issues["cand_missing_config_for_health"]))

    def test_plan_skips_quarantined_candidate(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_bad",
                "wired": True,
                "runner_type": "config_only",
                "entrypoint": "recbole.model.general_recommender.bpr:BPR",
            },
            {
                "candidate_id": "cand_good",
                "wired": True,
                "runner_type": "config_only",
                "entrypoint": "recbole.model.general_recommender.lightgcn:LightGCN",
            },
        ]
        rec_agent.quarantined_candidate_ids = {"cand_bad"}
        chosen, _, _ = rec_agent.plan()
        self.assertEqual(chosen["candidate_id"], "cand_good")


if __name__ == "__main__":
    unittest.main()
