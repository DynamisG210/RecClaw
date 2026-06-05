from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import importlib
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

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
    def test_custom_architecture_uses_nearest_parent_model_baseline(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_lightgcn_parent",
                "base_model": "LightGCN",
            },
            {
                "candidate_id": "cand_custom_child",
                "base_model": "Custom",
                "proposal_type": "architecture",
                "runnable_level": "architecture_required",
                "parent_candidate_id": "cand_lightgcn_parent",
            },
        ]
        rec_agent.baselines_by_model = {
            "LightGCN": {
                "model": "LightGCN",
                "status": "success",
                "ndcg@10": 0.2671,
            }
        }
        candidate = rec_agent.registry[1]
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "model": "CustomGraphRec",
                        "status": "success",
                        "ndcg@10": 0.271,
                    }
                ),
                encoding="utf-8",
            )
            _, compare_baseline, _, _ = rec_agent.evaluate(
                candidate,
                {
                    "summary": {
                        "run_id": "candidate_custom",
                        "model": "CustomGraphRec",
                        "result_json_path": str(result_path),
                    }
                },
            )
        self.assertEqual(compare_baseline["baseline_metric"], 0.2671)
        self.assertEqual(compare_baseline["decision"], "keep")

    def test_architecture_family_is_distinct_from_lineage_parent(self) -> None:
        candidate = {
            "candidate_id": "cand_custom_dual_path",
            "parent_candidate_id": "cand_lightgcn_edge_dropout_residual_norm",
            "base_model": "Custom",
            "proposal_type": "architecture",
            "runnable_level": "architecture_required",
            "action_type": "propagation_mechanism",
            "base_architecture": "graph_propagation",
            "mechanism": "dual_path_alignment",
        }
        family = agent.RecClawAgent._candidate_family(candidate)
        self.assertEqual(family, "architecture::graph_propagation")
        self.assertEqual(
            agent.RecClawAgent._family_key(
                {
                    "family": family,
                    "parent_candidate_id": candidate["parent_candidate_id"],
                    "candidate_id": candidate["candidate_id"],
                }
            ),
            family,
        )
        self.assertEqual(
            experience.candidate_family(
                {
                    "family": family,
                    "parent_candidate_id": candidate["parent_candidate_id"],
                    "candidate_id": candidate["candidate_id"],
                }
            ),
            family,
        )
        self.assertEqual(experience.infer_base_model({"base_model": "Custom", **candidate}), "Custom")

    def test_seed_validation_prefers_registered_custom_candidate_over_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                    state_summary_path=Path(tmp) / "state.json",
                )
            )
            rec_agent.registry = [
                {
                    "candidate_id": "cand_lightgcn_parent",
                    "base_model": "LightGCN",
                    "entrypoint": "recbole.model.general_recommender.lightgcn:LightGCN",
                },
                {
                    "candidate_id": "cand_custom_child",
                    "base_model": "Custom",
                    "proposal_type": "architecture",
                    "runnable_level": "architecture_required",
                    "parent_candidate_id": "cand_lightgcn_parent",
                    "entrypoint": "recclaw_ext.models.custom_child:CustomChild",
                },
            ]
            rec_agent.memory = [
                {
                    "round_id": 1,
                    "candidate_id": "cand_custom_child",
                    "parent_candidate_id": "cand_lightgcn_parent",
                    "decision": "keep",
                    "params": {"embedding_size": 64},
                    "compare_baseline": {"baseline_metric": 0.2671},
                    "result": {"ndcg@10": 0.27},
                }
            ]
            observed: dict[str, object] = {}

            def fake_validation(candidate, params, baseline_metric):
                observed["candidate"] = candidate
                observed["params"] = params
                observed["baseline_metric"] = baseline_metric
                return {"status": "passed"}

            rec_agent.run_seed_validation = fake_validation
            rec_agent.verify_last_keep(round_id=2)

        selected = observed["candidate"]
        self.assertEqual(selected["candidate_id"], "cand_custom_child")
        self.assertEqual(selected["base_model"], "Custom")
        self.assertEqual(selected["entrypoint"], "recclaw_ext.models.custom_child:CustomChild")
        self.assertNotIn("run_candidate_id", selected)

    def test_loop_mode_policies_cover_requested_modes(self) -> None:
        self.assertEqual(set(agent.LOOP_MODE_POLICIES), {"tuning", "mixed", "explore", "auto"})
        self.assertEqual(agent.LOOP_MODE_POLICIES["tuning"]["proposal_mode"], "conservative")
        self.assertTrue(agent.LOOP_MODE_POLICIES["explore"]["auto_implement"])
        self.assertEqual(agent.LOOP_MODE_POLICIES["auto"]["proposal_mode"], "algorithm_first")
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
        locally_declared = {
            str(item.get("name"))
            for candidate in registry.get("candidates", [])
            for item in (candidate.get("new_parameters") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        ignored = {"train_neg_sample_args"}
        missing_from_space = consumed - set(agent.AgentConfig().parameter_space) - locally_declared - ignored
        missing_from_schema = consumed - set(agent.PROPOSAL_PARAMETER_KEYS) - locally_declared - ignored
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
        priority_families.update(policy["search_stages"]["algorithm_discovery"]["priority_families"])
        self.assertEqual(policy["protocol_lock"]["primary_metric"], "ndcg@10")
        self.assertEqual(policy["family_budget"]["max_code_required_per_window"], 3)
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

    def test_new_candidate_entrypoints_are_importable_when_recbole_is_available(self) -> None:
        registry = agent.load_yaml(ROOT / "configs" / "candidate_registry.yaml")
        target_ids = {
            "cand_bpr_hard_negative_mix",
            "cand_bpr_hard_negative_margin",
            "cand_bpr_long_tail_reweight",
            "cand_bpr_norm_constrained",
            "cand_bpr_popularity_aware_negative",
            "cand_bpr_popularity_aware_margin",
            "cand_bpr_popularity_regularized",
            "cand_lightgcn_edge_dropout_residual_mix",
            "cand_lightgcn_edge_dropout_residual_norm",
            "cand_lightgcn_edge_dropout_residual_norm_dualpathblend_repair_076",
            "cand_lightgcn_debiased_negative_sampling",
            "cand_lightgcn_aux_alignment_loss",
            "cand_lightgcn_rank_aware_loss",
            "cand_lightgcn_residual_norm_constrained",
        }
        rows = {
            str(item.get("candidate_id")): item
            for item in registry.get("candidates", [])
            if str(item.get("candidate_id")) in target_ids
        }
        self.assertEqual(set(rows), target_ids)
        for candidate_id, row in rows.items():
            module_name, attr = str(row["entrypoint"]).split(":", 1)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name in {"recbole", "torch"}:
                    self.skipTest("RecBole is not installed in this local test environment")
                raise
            self.assertTrue(hasattr(module, attr), candidate_id)

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

    def test_auto_planner_sanitizes_empty_action_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    loop_mode="auto",
                    allow_llm_fallback=False,
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                    state_summary_path=Path(tmp) / "agent_state_summary.json",
                )
            )

            def empty_action(*_args: object, **_kwargs: object) -> str:
                return '{"action": "", "reason": "blank from provider", "proposal_count": 1}'

            rec_agent._chat_completion = empty_action  # type: ignore[method-assign]
            rec_agent.apply_auto_planner(round_id=1)
            self.assertEqual(rec_agent.last_planner_action["action"], "propose_mixed")
            self.assertTrue(rec_agent.force_proposal_refresh)
            self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_auto_round_policy_resets_mutated_explore_state(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig(loop_mode="auto"))
        rec_agent.config.proposal_mode = "explore"
        rec_agent.config.enable_candidate_proposals = False
        rec_agent.config.auto_implement_code_required = False
        rec_agent.reset_round_policy()
        self.assertEqual(rec_agent.config.proposal_mode, "algorithm_first")
        self.assertTrue(rec_agent.config.enable_candidate_proposals)
        self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_auto_round_policy_preserves_explicit_proposal_mode(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                loop_mode="auto",
                proposal_mode="architecture_first",
                proposal_mode_locked=True,
            )
        )
        rec_agent.config.enable_candidate_proposals = False
        rec_agent.config.auto_implement_code_required = False
        rec_agent.reset_round_policy()
        self.assertEqual(rec_agent.config.proposal_mode, "architecture_first")
        self.assertTrue(rec_agent.config.enable_candidate_proposals)
        self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_locked_architecture_mode_only_plans_architecture_candidates(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                proposal_mode="architecture_first",
                proposal_mode_locked=True,
            )
        )
        rec_agent.registry = [
            {
                "candidate_id": "cand_legacy_bpr",
                "base_model": "BPR",
                "runner_type": "model",
                "wired": True,
                "status": "implemented",
                "consumes": [],
            },
            {
                "candidate_id": "cand_custom_attention",
                "base_model": "Custom",
                "proposal_type": "architecture",
                "action_type": "graph_attention",
                "base_architecture": "graph_attention",
                "runnable_level": "architecture_required",
                "runner_type": "model",
                "wired": True,
                "status": "implemented",
                "consumes": [],
            },
        ]
        candidate, _, _ = rec_agent.plan()
        self.assertEqual(candidate["candidate_id"], "cand_custom_attention")

    def test_locked_architecture_mode_refuses_legacy_only_budget(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                proposal_mode="architecture_first",
                proposal_mode_locked=True,
            )
        )
        rec_agent.registry = [
            {
                "candidate_id": "cand_legacy_lightgcn",
                "base_model": "LightGCN",
                "runner_type": "model",
                "wired": True,
                "status": "implemented",
                "consumes": [],
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "locked proposal mode architecture_first"):
            rec_agent.plan()

    def test_auto_planner_preserves_explicit_architecture_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    loop_mode="auto",
                    proposal_mode="architecture_first",
                    proposal_mode_locked=True,
                    allow_llm_fallback=False,
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                    state_summary_path=Path(tmp) / "agent_state_summary.json",
                )
            )

            def algorithm_action(*_args: object, **_kwargs: object) -> str:
                return '{"action": "propose_algorithm", "reason": "planner default", "proposal_count": 1}'

            rec_agent._chat_completion = algorithm_action  # type: ignore[method-assign]
            rec_agent.apply_auto_planner(round_id=1)
            self.assertEqual(rec_agent.last_planner_action["action"], "propose_algorithm")
            self.assertEqual(rec_agent.config.proposal_mode, "architecture_first")
            self.assertTrue(rec_agent.force_proposal_refresh)
            self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_algorithm_first_planner_fallback_prefers_algorithm_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    loop_mode="auto",
                    search_intensity="algorithm_first",
                    allow_llm_fallback=True,
                    memory_path=Path(tmp) / "agent_memory.jsonl",
                    state_summary_path=Path(tmp) / "agent_state_summary.json",
                )
            )

            def fail_chat(*_args: object, **_kwargs: object) -> str:
                raise RuntimeError("api down")

            rec_agent._chat_completion = fail_chat  # type: ignore[method-assign]
            rec_agent.apply_auto_planner(round_id=1)
            self.assertEqual(rec_agent.last_planner_action["action"], "propose_algorithm")
            self.assertEqual(rec_agent.config.proposal_mode, "algorithm_first")
            self.assertTrue(rec_agent.force_proposal_refresh)
            self.assertTrue(rec_agent.config.auto_implement_code_required)

    def test_anti_plateau_overrides_local_tuning_action(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                search_intensity="algorithm_first",
                plateau_window_metric_rows=10,
                plateau_family_overuse_window=10,
                max_same_family_repair_streak=4,
            )
        )
        rec_agent.memory = [
            {
                "round_id": 1,
                "candidate_id": "cand_global_best",
                "parent_candidate_id": "cand_lightgcn_residual_norm_constrained",
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.2842},
            }
        ]
        for round_id in range(2, 18):
            rec_agent.memory.append(
                {
                    "round_id": round_id,
                    "candidate_id": f"cand_local_repair_{round_id}",
                    "parent_candidate_id": "cand_lightgcn_shallow_gate_plateau",
                    "status": "success",
                    "decision": "revise",
                    "result": {"ndcg@10": 0.277},
                }
            )
        rec_agent.memory.append(
            {
                "event": "proposal_needs_review",
                "proposal_id": "cand_stale_local_code",
                "parent_candidate_id": "cand_lightgcn_shallow_gate_plateau",
                "next_action": "promote_to_implementation_queue",
            }
        )

        payload, reason = rec_agent._maybe_override_auto_action(
            {"action": "tune_after_algorithm_success", "reason": "keep repairing gate", "proposal_count": 3}
        )

        self.assertEqual(payload["action"], "propose_objective")
        self.assertEqual(reason, "anti_plateau_force_algorithm_exploration")
        self.assertTrue(rec_agent._plateau_state()["plateau_detected"])
        self.assertFalse(rec_agent._has_pending_code_required_review())

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

    def test_auto_experience_refresh_updates_directive_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "experience_summary.md"
            summary_path.write_text("initial summary", encoding="utf-8")
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(
                    use_experiment_directive=True,
                    experiment_directive="initial summary",
                    refresh_experience_every=10,
                    memory_path=root / "agent_memory.jsonl",
                    state_summary_path=root / "agent_state_summary.json",
                    results_csv=root / "results.csv",
                    proposal_path=root / "candidate_proposals.jsonl",
                    candidate_tree_path=root / "candidate_search_tree.json",
                    candidate_tree_md_path=root / "candidate_search_tree.md",
                    candidate_tree_mmd_path=root / "candidate_search_tree.mmd",
                    experience_summary_path=summary_path,
                    experience_summary_json_path=root / "experience_summary.json",
                    reflection_memory_path=root / "reflection_memory.jsonl",
                )
            )
            calls: list[list[str]] = []

            class Completed:
                returncode = 0
                stdout = ""
                stderr = ""

            def fake_run(cmd: list[str], **_kwargs: object) -> Completed:
                calls.append(cmd)
                if "build_experience_summary.py" in str(cmd[1]):
                    summary_path.write_text("updated summary", encoding="utf-8")
                return Completed()

            original_run = agent.subprocess.run
            agent.subprocess.run = fake_run  # type: ignore[assignment]
            try:
                refreshed = rec_agent._refresh_experience_artifacts(10, reason="unit_test")
            finally:
                agent.subprocess.run = original_run  # type: ignore[assignment]

            self.assertTrue(refreshed)
            self.assertEqual(len(calls), 2)
            self.assertEqual(rec_agent.config.experiment_directive, "updated summary")
            self.assertTrue(any(row.get("event") == "experience_refresh" for row in rec_agent.memory))

    def test_auto_experience_refresh_is_disabled_with_no_directive(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                use_experiment_directive=False,
                refresh_experience_every=10,
            )
        )
        self.assertFalse(rec_agent._refresh_experience_artifacts(10, reason="unit_test"))

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

    def test_experience_policy_removes_avoid_families_from_prefer_and_repair(self) -> None:
        summary = {
            "family_summaries": [
                {
                    "family": "cand_bad_default",
                    "policy_bucket": "avoid",
                    "crash_count": 0,
                    "collapse_rate": 0.0,
                    "decisions": {"discard": 2},
                    "best": 0.1,
                },
                {
                    "family": "cand_caution_frozen",
                    "policy_bucket": "caution",
                    "crash_count": 0,
                    "collapse_rate": 0.0,
                    "decisions": {"discard": 2},
                    "best": 0.27,
                },
            ],
            "failed_compositions": [],
            "do_not_repeat_signatures": [],
        }
        search_policy = {
            "family_budget": {"crashes_before_freeze": 2, "collapse_rate_before_freeze": 0.25},
            "search_stages": {"exploit": {"priority_families": ["cand_bad_default", "cand_good_default"]}},
        }
        policy = experience.build_experience_policy(summary, search_policy=search_policy, tree_summary={})
        next_policy = policy["next_proposal_policy"]
        self.assertEqual(next_policy["prefer_families"], ["cand_good_default"])
        self.assertNotIn("cand_bad_default", next_policy["prefer_families"])
        self.assertNotIn("cand_caution_frozen", next_policy["repair_families"])
        self.assertIn("cand_bad_default", next_policy["avoid_families"])
        self.assertIn("cand_caution_frozen", next_policy["avoid_families"])

    def test_experience_policy_forces_cross_family_after_plateau(self) -> None:
        search_policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        reflection_policy = experience.load_yaml(ROOT / "configs" / "reflection_policy.yaml")
        rows = [
            {
                "round_id": 1,
                "candidate_id": "cand_best",
                "parent_candidate_id": "cand_lightgcn_residual_norm_constrained",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "keep",
                "params": {"lambda_norm": 0.001},
                "result": {"ndcg@10": 0.2842},
            }
        ]
        for round_id in range(2, 40):
            rows.append(
                {
                    "round_id": round_id,
                    "candidate_id": f"cand_plateau_{round_id}",
                    "parent_candidate_id": "cand_lightgcn_shallow_gate_plateau",
                    "base_model": "LightGCN",
                    "status": "success",
                    "decision": "revise",
                    "params": {"lambda_align": 0.01},
                    "result": {"ndcg@10": 0.277},
                }
            )

        summary = experience.summarize_memory(
            rows,
            action_space={},
            policy=reflection_policy,
            search_policy=search_policy,
        )
        policy = experience.build_experience_policy(summary, search_policy=search_policy, tree_summary={})
        next_policy = policy["next_proposal_policy"]

        self.assertTrue(policy["plateau_analysis"]["plateau_detected"])
        self.assertIn("cand_lightgcn_shallow_gate", next_policy["local_repair_capped_families"])
        self.assertIn("cand_lightgcn_residual_norm_constrained", next_policy["forced_exploration_families"])
        self.assertNotIn("cand_lightgcn_shallow_gate_plateau", next_policy["prefer_families"])

    def test_high_potential_anchor_goes_to_revisit_not_freeze(self) -> None:
        search_policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        reflection_policy = experience.load_yaml(ROOT / "configs" / "reflection_policy.yaml")
        rows = [
            {
                "candidate_id": "cand_lightgcn_residual_norm_constrained_trial_a",
                "parent_candidate_id": "cand_lightgcn_residual_norm_constrained",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "discard",
                "params": {"lambda_norm": 0.001, "max_norm": 1.0},
                "result": {"ndcg@10": 0.265},
            },
            {
                "candidate_id": "cand_lightgcn_residual_norm_constrained_trial_b",
                "parent_candidate_id": "cand_lightgcn_residual_norm_constrained",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "discard",
                "params": {"lambda_norm": 0.0005, "max_norm": 0.5},
                "result": {"ndcg@10": 0.266},
            },
        ]
        summary = experience.summarize_memory(
            rows,
            action_space={},
            policy=reflection_policy,
            search_policy=search_policy,
        )
        family = summary["family_summaries"][0]
        self.assertEqual(family["policy_bucket"], "revisit")
        policy = experience.build_experience_policy(summary, search_policy=search_policy, tree_summary={})
        self.assertIn("cand_lightgcn_residual_norm_constrained", policy["revisit_families"])
        self.assertNotIn("cand_lightgcn_residual_norm_constrained", policy["avoid_families"])
        self.assertNotIn("cand_lightgcn_residual_norm_constrained", policy["freeze_families"])

    def test_experience_builder_outputs_domain_and_trend_notes(self) -> None:
        search_policy = experience.load_yaml(ROOT / "configs" / "search_policy.yaml")
        reflection_policy = experience.load_yaml(ROOT / "configs" / "reflection_policy.yaml")
        rows = [
            {
                "round_id": 1,
                "candidate_id": "cand_lightgcn_deep_trial",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "revise",
                "params": {"n_layers": 4},
                "result": {"ndcg@10": 0.268},
            },
            {
                "round_id": 2,
                "candidate_id": "cand_lightgcn_deep_trial",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "revise",
                "params": {"n_layers": 4},
                "result": {"ndcg@10": 0.27},
            },
            {
                "round_id": 3,
                "candidate_id": "cand_lightgcn_deep_trial",
                "base_model": "LightGCN",
                "status": "success",
                "decision": "revise",
                "params": {"n_layers": 4},
                "result": {"ndcg@10": 0.272},
            },
        ]
        summary = experience.summarize_memory(
            rows,
            action_space={},
            policy=reflection_policy,
            search_policy=search_policy,
        )
        family = summary["family_summaries"][0]
        self.assertEqual(family["trend"]["trend"], "improving")
        self.assertTrue(family["domain_warnings"])
        policy = experience.build_experience_policy(summary, search_policy=search_policy, tree_summary={})
        self.assertTrue(policy["domain_prior_notes"])
        self.assertTrue(policy["trend_notes"])

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

    def test_template_implementation_maps_known_hard_negative_margin(self) -> None:
        proposal = {
            "candidate_id": "cand_bpr_hard_negative_margin_generated",
            "parent_candidate_id": "cand_bpr_hard_negative_mix",
            "base_model": "BPR",
            "category": "Bias & Sample Construction",
            "hypothesis": "Try hard negatives with a small margin.",
            "runner_type": "model",
            "consumes": ["hard_negative_ratio", "margin"],
            "parameter_overrides": {"hard_negative_ratio": 0.25, "margin": 0.1},
        }

        prepared = implement.template_implementation_for_proposal(proposal)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.files, [])
        self.assertEqual(prepared.source, "template:bpr_hard_negative_margin")
        self.assertEqual(prepared.entrypoint, "recclaw_ext.models.bpr_composed:BPRHardNegativeMargin")
        self.assertEqual(prepared.candidate_config["hard_negative_ratio"], 0.25)
        self.assertEqual(prepared.candidate_config["margin"], 0.1)

    def test_template_implementation_prefers_edge_dropout_residual_norm(self) -> None:
        proposal = {
            "candidate_id": "cand_lightgcn_edge_dropout_residual_norm_generated",
            "parent_candidate_id": "cand_lightgcn_edge_dropout_residual_mix",
            "base_model": "LightGCN",
            "category": "Representation & Interaction",
            "hypothesis": "Compose edge dropout with norm control.",
            "runner_type": "model",
            "consumes": [
                "embedding_size",
                "n_layers",
                "residual_weight",
                "edge_dropout",
                "lambda_norm",
                "max_norm",
            ],
            "parameter_overrides": {"edge_dropout": 0.05, "lambda_norm": 1e-5, "max_norm": 2.0},
        }

        prepared = implement.template_implementation_for_proposal(proposal)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.source, "template:lightgcn_edge_dropout_residual_norm")
        self.assertEqual(
            prepared.entrypoint,
            "recclaw_ext.models.lightgcn_residual_norm:LightGCNEdgeDropoutResidualNorm",
        )
        self.assertEqual(prepared.candidate_config["edge_dropout"], 0.05)
        self.assertEqual(prepared.candidate_config["lambda_norm"], 1e-5)
        self.assertEqual(prepared.candidate_config["max_norm"], 2.0)

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

    def test_auto_implementation_rejects_direct_float_config_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "config_float"):
            implement.validate_static_model_code(
                "recclaw_ext/models/generated_bad_config.py",
                "class X:\n"
                "    def __init__(self, config):\n"
                "        self.lambda_align = float(config['lambda_align'])\n",
            )

    def test_candidate_config_requires_consumed_defaults(self) -> None:
        with self.assertRaisesRegex(ValueError, "lambda_align"):
            implement.validate_candidate_config_matches_proposal(
                {"candidate_id": "cand_x", "model": "X"},
                {
                    "candidate_id": "cand_x",
                    "consumes": ["lambda_align"],
                    "parameter_overrides": {},
                },
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

    def test_seed_verify_pending_rows_skip_separate_validation_events(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.memory = [
            {
                "round_id": 1,
                "candidate_id": "cand_verified",
                "parent_candidate_id": "parent_a",
                "params": {"x": 1},
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.28},
                "parameter_signature": 'parent_a::{"x":1}',
            },
            {
                "event": "seed_validation",
                "round_id": 2,
                "candidate_id": "cand_verified",
                "parent_candidate_id": "parent_a",
                "seed_validation": {"status": "passed", "mean": 0.281},
            },
            {
                "round_id": 3,
                "candidate_id": "cand_pending",
                "parent_candidate_id": "parent_a",
                "params": {"x": 2},
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.282},
                "parameter_signature": 'parent_a::{"x":2}',
            },
        ]

        pending = rec_agent._unverified_keep_rows()

        self.assertEqual([row["candidate_id"] for row in pending], ["cand_pending"])

    def test_best_validated_keep_uses_separate_validation_events(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.memory = [
            {
                "round_id": 1,
                "candidate_id": "cand_verified",
                "parent_candidate_id": "parent_a",
                "params": {"x": 1},
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.28},
                "parameter_signature": 'parent_a::{"x":1}',
            },
            {
                "event": "seed_validation",
                "round_id": 2,
                "candidate_id": "cand_verified",
                "parent_candidate_id": "parent_a",
                "parameter_signature": 'parent_a::{"x":1}',
                "seed_validation": {"status": "passed", "mean": 0.281},
            },
        ]

        best = rec_agent._best_validated_keep()

        self.assertEqual(best["candidate_id"], "cand_verified")
        self.assertEqual(best["seed_validation"]["mean"], 0.281)

    def test_failed_seed_validation_clears_pending_keep_gate(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.memory = [
            {
                "round_id": 1,
                "candidate_id": "cand_failed",
                "parent_candidate_id": "parent_a",
                "params": {"x": 1},
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.28},
                "parameter_signature": 'parent_a::{"x":1}',
            },
            {
                "event": "seed_validation",
                "round_id": 2,
                "candidate_id": "cand_failed",
                "parent_candidate_id": "parent_a",
                "parameter_signature": 'parent_a::{"x":1}',
                "seed_validation": {"status": "failed", "mean": 0.25},
            },
        ]

        self.assertEqual(rec_agent._unverified_keep_rows(), [])
        self.assertEqual(rec_agent._best_validated_keep(), {})

    def test_run_continuation_respects_start_round(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                start_round=3,
                rounds=4,
                dry_run=True,
                enable_candidate_proposals=False,
            )
        )
        seen_rounds: list[int] = []
        refreshes: list[tuple[int, str]] = []
        rec_agent.observe = lambda: None  # type: ignore[method-assign]
        rec_agent.remember_experiment_directive = lambda: None  # type: ignore[method-assign]
        rec_agent.reset_round_policy = lambda: None  # type: ignore[method-assign]
        rec_agent._refresh_experience_artifacts = (  # type: ignore[method-assign]
            lambda round_id, reason: refreshes.append((round_id, reason)) or True
        )

        def fake_apply(round_id: int) -> None:
            seen_rounds.append(round_id)
            rec_agent.skip_current_round = False

        rec_agent.apply_auto_planner = fake_apply  # type: ignore[method-assign]
        rec_agent.plan = lambda: ({"candidate_id": f"cand_{len(seen_rounds)}"}, {}, {})  # type: ignore[method-assign]

        rec_agent.run()

        self.assertEqual(seen_rounds, [3, 4])
        self.assertEqual(refreshes, [])

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

    def test_architecture_protocol_accepts_general_recommender_contract(self) -> None:
        files = [
            {
                "path": "recclaw_ext/models/architecture_exploration.py",
                "content": (
                    "class CustomGraphAttention(GeneralRecommender):\n"
                    "    input_type = InputType.PAIRWISE\n"
                    "    def calculate_loss(self, interaction):\n"
                    "        return self.some_scalar_loss\n"
                    "    def predict(self, interaction):\n"
                    "        return self.some_score\n"
                    "    def full_sort_predict(self, interaction):\n"
                    "        return self.some_full_sort_score\n"
                ),
            }
        ]

        implement.validate_architecture_protocol(
            files,
            "recclaw_ext.models.architecture_exploration:CustomGraphAttention",
        )

    def test_architecture_protocol_rejects_string_input_type(self) -> None:
        files = [
            {
                "path": "recclaw_ext/models/architecture_exploration.py",
                "content": (
                    "class CustomGraphAttention(GeneralRecommender):\n"
                    "    input_type = 'pairwise'\n"
                    "    def calculate_loss(self, interaction):\n"
                    "        return self.some_scalar_loss\n"
                    "    def predict(self, interaction):\n"
                    "        return self.some_score\n"
                    "    def full_sort_predict(self, interaction):\n"
                    "        return self.some_full_sort_score\n"
                ),
            }
        ]

        with self.assertRaisesRegex(ValueError, "InputType enum"):
            implement.validate_architecture_protocol(
                files,
                "recclaw_ext.models.architecture_exploration:CustomGraphAttention",
            )

    def test_prepare_implementation_preserves_proposal_lineage_in_registry(self) -> None:
        candidate_id = "cand_custom_registry_lineage_test"
        implementation = {
            "files": [
                {
                    "path": "recclaw_ext/models/custom_registry_lineage_test.py",
                    "content": (
                        "class RegistryLineageModel(GeneralRecommender):\n"
                        "    input_type = InputType.PAIRWISE\n"
                        "    def calculate_loss(self, interaction):\n"
                        "        return self.some_scalar_loss\n"
                        "    def predict(self, interaction):\n"
                        "        return self.some_score\n"
                        "    def full_sort_predict(self, interaction):\n"
                        "        return self.some_full_sort_score\n"
                    ),
                }
            ],
            "candidate_config": {
                "candidate_id": candidate_id,
                "model": "RegistryLineageModel",
                "architecture_gate_alpha": 0.2,
            },
            "registry_entry": {
                "candidate_id": candidate_id,
                "category": "Architecture",
                "base_model": "Custom",
                "rs_problem": "test",
                "hypothesis": "test",
                "implementation_type": "model",
                "minimal_change": "test",
                "priority": "high",
                "status": "implemented",
                "wired": True,
                "runner_type": "model",
                "entrypoint": "recclaw_ext.models.custom_registry_lineage_test:RegistryLineageModel",
                "consumes": ["architecture_gate_alpha"],
            },
        }
        proposal = {
            "candidate_id": candidate_id,
            "proposal_type": "architecture",
            "action_type": "graph_attention",
            "base_architecture": "graph_attention",
            "runnable_level": "architecture_required",
            "parent_candidate_id": "cand_lightgcn_parent",
            "ablation_parent": "cand_lightgcn_parent",
            "mechanism": "registry_lineage_test",
            "mechanism_composition": ["graph_attention"],
            "parameter_overrides": {},
            "new_parameters": [
                {
                    "name": "architecture_gate_alpha",
                    "type": "number",
                    "default": 0.2,
                    "search_space": [0.1, 0.2, 0.5],
                }
            ],
            "consumes": ["architecture_gate_alpha"],
        }
        prepared = implement.prepare_implementation(implementation, proposal)
        self.assertEqual(prepared.registry_entry["proposal_type"], "architecture")
        self.assertEqual(prepared.registry_entry["action_type"], "graph_attention")
        self.assertEqual(prepared.registry_entry["base_architecture"], "graph_attention")
        self.assertEqual(prepared.registry_entry["runnable_level"], "architecture_required")
        self.assertEqual(prepared.registry_entry["parent_candidate_id"], "cand_lightgcn_parent")
        self.assertEqual(
            prepared.registry_entry["new_parameters"][0]["name"],
            "architecture_gate_alpha",
        )

    def test_candidate_config_params_include_declared_local_parameters(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        candidate_id = "cand_custom_local_parameter_test"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs" / "candidates"
            config_dir.mkdir(parents=True)
            (config_dir / f"{candidate_id}.yaml").write_text(
                (
                    f"candidate_id: {candidate_id}\n"
                    "model: LocalModel\n"
                    "embedding_size: 64\n"
                    "architecture_gate_alpha: 0.2\n"
                ),
                encoding="utf-8",
            )
            candidate = {
                "candidate_id": candidate_id,
                "consumes": ["embedding_size", "architecture_gate_alpha"],
                "new_parameters": [
                    {
                        "name": "architecture_gate_alpha",
                        "default": 0.2,
                        "search_space": [0.1, 0.2, 0.5],
                    }
                ],
            }
            with mock.patch.object(agent, "PROJECT_ROOT", root):
                params = rec_agent._candidate_config_params(candidate)

        self.assertEqual(
            params,
            {"embedding_size": 64, "architecture_gate_alpha": 0.2},
        )
        self.assertEqual(
            rec_agent._candidate_parameter_values(candidate, "architecture_gate_alpha"),
            [0.1, 0.2, 0.5],
        )

    def test_architecture_protocol_rejects_parent_model_subclass(self) -> None:
        files = [
            {
                "path": "recclaw_ext/models/architecture_exploration.py",
                "content": (
                    "class CustomGraphAttention(LightGCN):\n"
                    "    def calculate_loss(self, interaction):\n"
                    "        return self.some_scalar_loss\n"
                    "    def predict(self, interaction):\n"
                    "        return self.some_score\n"
                    "    def full_sort_predict(self, interaction):\n"
                    "        return self.some_full_sort_score\n"
                ),
            }
        ]

        with self.assertRaisesRegex(ValueError, "not BPR/LightGCN"):
            implement.validate_architecture_protocol(
                files,
                "recclaw_ext.models.architecture_exploration:CustomGraphAttention",
            )

    def test_architecture_protocol_rejects_raw_torch_module(self) -> None:
        files = [
            {
                "path": "recclaw_ext/models/architecture_exploration.py",
                "content": (
                    "class CustomGraphAttention(nn.Module):\n"
                    "    def calculate_loss(self, interaction):\n"
                    "        return self.some_scalar_loss\n"
                    "    def predict(self, interaction):\n"
                    "        return self.some_score\n"
                    "    def full_sort_predict(self, interaction):\n"
                    "        return self.some_full_sort_score\n"
                ),
            }
        ]

        with self.assertRaisesRegex(ValueError, "raw torch Module"):
            implement.validate_architecture_protocol(
                files,
                "recclaw_ext.models.architecture_exploration:CustomGraphAttention",
            )

    def test_implementation_schema_keeps_strict_candidate_config_with_new_params(self) -> None:
        schema = implement.implementation_response_schema_for_proposal(
            {
                "new_parameters": [
                    {
                        "name": "architecture_gate_alpha",
                        "type": "number",
                        "default": 0.2,
                        "search_space": [0.1, 0.2, 0.5],
                    },
                    {
                        "name": "gate_mode",
                        "type": "string",
                        "default": "residual",
                        "search_space": ["residual", "direct"],
                    },
                ]
            }
        )
        candidate_schema = schema["properties"]["candidate_config"]

        self.assertFalse(candidate_schema["additionalProperties"])
        self.assertIn("architecture_gate_alpha", candidate_schema["properties"])
        self.assertIn("gate_mode", candidate_schema["properties"])
        self.assertIn("architecture_gate_alpha", candidate_schema["required"])
        self.assertIn("gate_mode", candidate_schema["required"])
        self.assertEqual(candidate_schema["properties"]["gate_mode"]["type"], ["string", "null"])

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
        summary = implement.compact_smoke_summary(
            completed,
            ["epochs=3", "stopping_step=2"],
            candidate_config={
                "candidate_id": "cand_x",
                "model": "X",
                "architecture_gate_alpha": 0.2,
            },
        )
        self.assertTrue(summary["metric_stagnation"])
        self.assertEqual(summary["valid_score_unique_count"], 1)
        self.assertEqual(summary["run_id"], "r1")
        self.assertEqual(summary["candidate_parameters"], {"architecture_gate_alpha": 0.2})

    def test_auto_implementation_stops_after_repeated_smoke_failures(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig(max_failed_implementation_attempts=2))
        rec_agent.memory = [
            {
                "event": "implementation_result",
                "proposal_id": "cand_bad",
                "candidate_id": "cand_bad",
                "status": "implemented_but_smoke_failed",
            },
            {
                "event": "implementation_result",
                "proposal_id": "cand_bad",
                "candidate_id": "cand_bad",
                "status": "implemented_but_smoke_failed",
            },
        ]
        rec_agent.proposal_validation_report = {
            "results": [
                {
                    "candidate_id": "cand_bad",
                    "status": "needs_review",
                    "runnable_level": "code_required",
                }
            ]
        }
        rec_agent.candidate_proposals = [{"candidate_id": "cand_bad"}]
        rec_agent._run_json_command = lambda *args, **kwargs: self.fail("implementation should be skipped")  # type: ignore[method-assign]

        rec_agent.implement_needs_review_proposals(round_id=3)

        self.assertEqual(rec_agent.memory[-1]["event"], "implementation_skipped")
        self.assertEqual(rec_agent.memory[-1]["reason"], "prior_failed_implementation_attempts")

    def test_algorithm_first_keep_requires_promotion_floor(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(search_intensity="algorithm_first", seed_validation_min_metric=0.274)
        )

        decision, reason, _ = rec_agent.reflect(
            {"candidate_id": "cand_bpr_margin_loss", "consumes": ["margin"]},
            {"margin": 0.2},
            {"exit_code": 0},
            {"status": "success", "ndcg@10": 0.259},
            {"delta": 0.03},
            {"delta": 0.001},
        )

        self.assertEqual(decision, "revise")
        self.assertIn("promotion floor", reason)

    def test_proposal_route_records_algorithm_metadata(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.current_proposal_source = "llm"
        rec_agent.candidate_proposals = [
            {
                "candidate_id": "cand_alg",
                "parent_candidate_id": "cand_parent",
                "action_type": "auxiliary_loss",
                "proposal_type": "algorithmic_variant",
                "runnable_level": "code_required",
                "mechanism_composition": ["alignment", "rank-aware"],
                "novelty_claim": "minimal mechanism ablation",
                "expected_failure_mode": "loss scale instability",
                "ablation_parent": "cand_parent",
                "implementation_complexity": "low",
            }
        ]
        rec_agent.proposal_validation_report = {
            "results": [
                {
                    "candidate_id": "cand_alg",
                    "status": "needs_review",
                    "parameter_signature": "",
                    "errors": [],
                    "review_reasons": [],
                    "next_action": "promote_to_implementation_queue",
                }
            ]
        }

        rec_agent.record_proposal_routes(round_id=7)

        event = rec_agent.memory[-1]
        self.assertEqual(event["mechanism"], "alignment+rank-aware")
        self.assertEqual(event["proposal_type"], "algorithmic_variant")
        self.assertEqual(event["runnable_level"], "code_required")

    def test_smoke_results_csv_is_isolated_from_runtime_results(self) -> None:
        old_results = os.environ.get("RECCLAW_RESULTS_CSV")
        old_smoke = os.environ.get("RECCLAW_SMOKE_RESULTS_CSV")
        try:
            os.environ.pop("RECCLAW_SMOKE_RESULTS_CSV", None)
            os.environ["RECCLAW_RESULTS_CSV"] = "/tmp/recclaw_runtime/results.csv"
            self.assertEqual(
                implement.default_smoke_results_csv(),
                Path("/tmp/recclaw_runtime/results.smoke.csv"),
            )
            os.environ["RECCLAW_SMOKE_RESULTS_CSV"] = "/tmp/recclaw_smoke/custom.csv"
            self.assertEqual(
                implement.default_smoke_results_csv(),
                Path("/tmp/recclaw_smoke/custom.csv"),
            )
        finally:
            if old_results is None:
                os.environ.pop("RECCLAW_RESULTS_CSV", None)
            else:
                os.environ["RECCLAW_RESULTS_CSV"] = old_results
            if old_smoke is None:
                os.environ.pop("RECCLAW_SMOKE_RESULTS_CSV", None)
            else:
                os.environ["RECCLAW_SMOKE_RESULTS_CSV"] = old_smoke

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

    def test_auto_implementation_accepts_architecture_required_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_agent = agent.RecClawAgent(
                agent.AgentConfig(memory_path=Path(tmp) / "agent_memory.jsonl")
            )
            rec_agent.candidate_proposals = [
                {
                    "candidate_id": "cand_custom_architecture_proposal",
                    "proposal_type": "architecture",
                    "action_type": "graph_attention",
                    "base_architecture": "graph_attention",
                    "parent_candidate_id": "cand_lightgcn_parent",
                }
            ]
            rec_agent.proposal_validation_report = {
                "results": [
                    {
                        "candidate_id": "cand_custom_architecture_proposal",
                        "status": "needs_review",
                        "runnable_level": "architecture_required",
                    }
                ]
            }
            commands: list[list[str]] = []

            def fake_command(cmd: list[str], *_args: object, **_kwargs: object) -> dict[str, object]:
                commands.append(cmd)
                return {
                    "status": "implemented_and_smoke_passed",
                    "candidate_id": "cand_custom_architecture",
                    "entrypoint": "recclaw_ext.models.architecture_exploration:CustomGraphAttention",
                }

            rec_agent._run_json_command = fake_command  # type: ignore[method-assign]
            rec_agent.implement_needs_review_proposals(round_id=4)

            self.assertTrue(commands)
            self.assertIn("--proposal-id", commands[0])
            event = rec_agent.memory[-1]
            self.assertEqual(event["event"], "implementation_result")
            self.assertEqual(event["proposal_type"], "architecture")
            self.assertEqual(event["runnable_level"], "architecture_required")
            self.assertEqual(event["base_architecture"], "graph_attention")

    def test_v4_algorithm_templates_map_to_runnable_local_entrypoints(self) -> None:
        proposals = [
            {
                "candidate_id": "cand_bpr_tail_test",
                "parent_candidate_id": "cand_bpr_hard_negative_margin",
                "base_model": "BPR",
                "consumes": ["hard_negative_ratio", "margin", "tail_weight_alpha"],
            },
            {
                "candidate_id": "cand_bpr_rank_test",
                "parent_candidate_id": "cand_bpr_hard_negative_margin",
                "base_model": "BPR",
                "consumes": ["hard_negative_ratio", "margin", "rank_weight_alpha"],
            },
            {
                "candidate_id": "cand_lgcn_rank_align_test",
                "parent_candidate_id": "cand_lightgcn_residual_norm_constrained",
                "base_model": "LightGCN",
                "consumes": [
                    "embedding_size",
                    "n_layers",
                    "residual_weight",
                    "lambda_norm",
                    "max_norm",
                    "lambda_align",
                    "rank_weight_alpha",
                ],
            },
            {
                "candidate_id": "cand_lgcn_gate_test",
                "parent_candidate_id": "cand_lightgcn_edge_dropout_residual_norm",
                "base_model": "LightGCN",
                "consumes": [
                    "embedding_size",
                    "n_layers",
                    "residual_weight",
                    "edge_dropout",
                    "lambda_norm",
                    "max_norm",
                    "residual_gate_scale",
                    "gate_dropout",
                ],
            },
        ]
        prepared = [implement.template_implementation_for_proposal(item) for item in proposals]
        self.assertTrue(all(item is not None for item in prepared))
        for item in prepared:
            assert item is not None
            module_name, attr = item.entrypoint.split(":", 1)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name in {"recbole", "torch"}:
                    self.skipTest("RecBole is not installed in this local test environment")
                raise
            self.assertTrue(hasattr(module, attr), item.entrypoint)

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

    def test_post_validation_sibling_churn_prefers_structured_refinement(self) -> None:
        rec_agent = agent.RecClawAgent(
            agent.AgentConfig(
                search_intensity="algorithm_first",
                post_validation_sibling_churn_limit=2,
                post_validation_structured_followup_window=5,
                post_validation_min_followup_improvement=0.0005,
                seed_validation_min_metric=0.274,
            )
        )
        rec_agent.memory = [
            {
                "round_id": 10,
                "candidate_id": "cand_validated_best",
                "parent_candidate_id": "cand_anchor",
                "status": "success",
                "decision": "keep",
                "result": {"ndcg@10": 0.289},
                "seed_validation": {"status": "passed", "mean": 0.289},
            },
            {
                "round_id": 11,
                "candidate_id": "cand_validated_best_child_a",
                "parent_candidate_id": "cand_validated_best",
                "status": "success",
                "decision": "revise",
                "result": {"ndcg@10": 0.272},
            },
            {
                "round_id": 12,
                "candidate_id": "cand_validated_best_child_b",
                "parent_candidate_id": "cand_validated_best",
                "status": "success",
                "decision": "revise",
                "result": {"ndcg@10": 0.273},
            },
        ]

        followup = rec_agent._post_validation_followup_state()
        payload, reason = rec_agent._maybe_override_auto_action(
            {"action": "propose_algorithm", "reason": "keep making siblings", "proposal_count": 3}
        )

        self.assertTrue(followup["needs_structured_followup"])
        self.assertEqual(payload["action"], "tune_after_algorithm_success")
        self.assertEqual(reason, "post_validation_structured_followup")

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
        self.assertTrue(any(item.get("name") == "lambda_new" for item in proposal["new_parameters"]))

    def test_llm_proposal_repair_does_not_mark_known_action_space_params_as_new(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_parent",
                "consumes": ["embedding_size"],
            },
        ]
        proposal = {
            "candidate_id": "cand_architecture",
            "parent_candidate_id": "cand_parent",
            "runnable_level": "architecture_required",
            "proposal_type": "architecture",
            "action_type": "graph_attention",
            "consumes": ["embedding_size", "attention_heads", "architecture_gate_alpha"],
            "new_parameters": [
                {
                    "name": "attention_heads",
                    "type": "number",
                    "default": 1,
                    "search_space": [1, 2, 4],
                }
            ],
        }

        rec_agent._repair_llm_proposal_metadata(
            proposal,
            {"cand_parent": rec_agent.registry[0]},
            set(),
        )

        new_names = {item.get("name") for item in proposal["new_parameters"]}
        self.assertNotIn("attention_heads", new_names)
        self.assertIn("architecture_gate_alpha", new_names)

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
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        emb = torch.tensor([[3.0, 4.0]])
        legacy = module.soft_l2_norm_penalty(emb, 2.0)
        keyword = module.soft_l2_norm_penalty(emb, max_norm=2.0)
        self.assertAlmostEqual(float(legacy), float(keyword))

    def test_negative_samplers_can_avoid_padding_item_zero(self) -> None:
        try:
            import importlib.util
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local test environment")

        spec = importlib.util.spec_from_file_location(
            "recclaw_test_samplers",
            ROOT / "recclaw_ext" / "models" / "_samplers.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        popularity = torch.ones(8)
        mixed = module.MixedNegativeSampler(8, popularity=popularity, hard_negative_ratio=0.5, avoid_zero=True)
        pop = module.PopularityAwareNegativeSampler(popularity, alpha=0.5, avoid_zero=True)
        self.assertTrue(bool((mixed.sample((256,)) > 0).all()))
        self.assertTrue(bool((pop.sample((256,)) > 0).all()))

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
