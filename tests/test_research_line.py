from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import agent  # noqa: E402
import plan_research_line_comparison as comparison  # noqa: E402
import research_line  # noqa: E402


class ResearchLineTests(unittest.TestCase):
    def test_producer_directives_allocate_algorithm_first_roles(self) -> None:
        policy = research_line.load_yaml(ROOT / "configs" / "search_policy.yaml")
        memory = research_line.build_search_memory([], anchor_families=["cand_anchor"])

        directives = research_line.build_producer_directives(
            search_policy=policy,
            search_memory=memory,
            proposal_count=6,
            mode="algorithm_first",
        )

        roles = {item.role for item in directives}
        self.assertIn("mechanism_discovery", roles)
        self.assertIn("template_expansion", roles)
        self.assertIn("parameter_sanity", roles)
        self.assertEqual(sum(item.target_count for item in directives), 6)

    def test_router_downranks_repeated_parameter_signature(self) -> None:
        memory = research_line.build_search_memory(
            [
                {
                    "candidate_id": "proposal_old",
                    "parent_candidate_id": "cand_parent",
                    "status": "success",
                    "decision": "discard",
                    "parameter_signature": 'cand_parent::{"margin":0.2}',
                    "result": {"ndcg@10": 0.25},
                },
            ]
        )
        duplicate = {
            "proposal_type": "tuning",
            "candidate_id": "proposal_dup",
            "parent_candidate_id": "cand_parent",
            "action_type": "parameter_tuning",
            "runnable_level": "parameter_only",
            "parameter_signature": 'cand_parent::{"margin":0.2}',
            "implementation_complexity": "low",
        }
        novel = {
            "proposal_type": "algorithmic_variant",
            "candidate_id": "proposal_new",
            "parent_candidate_id": "cand_parent",
            "action_type": "rank_aware_loss",
            "runnable_level": "code_required",
            "mechanism_composition": ["rank", "hard_negative"],
            "implementation_complexity": "medium",
            "ablation_parent": "cand_parent",
        }

        ordered = research_line.order_proposals_by_route(
            [duplicate, novel],
            search_memory=memory,
            search_policy={},
            limit=2,
        )

        self.assertEqual(ordered[0]["candidate_id"], "proposal_new")
        self.assertIn("router", ordered[0]["research_line"])

    def test_agent_plan_attaches_research_route_decision(self) -> None:
        rec_agent = agent.RecClawAgent(agent.AgentConfig())
        rec_agent.registry = [
            {
                "candidate_id": "cand_a",
                "base_model": "BPR",
                "runner_type": "config_only",
                "wired": True,
                "status": "implemented",
                "priority": "low",
                "consumes": [],
            },
            {
                "candidate_id": "cand_b",
                "base_model": "LightGCN",
                "runner_type": "config_only",
                "wired": True,
                "status": "implemented",
                "priority": "high",
                "consumes": [],
            },
        ]
        rec_agent.search_memory = research_line.build_search_memory([])

        chosen, _, _ = rec_agent.plan()

        self.assertEqual(chosen["candidate_id"], "cand_b")
        self.assertIn("route_decision", chosen)
        self.assertIn("base_score", chosen["route_decision"]["factors"])

    def test_meta_research_update_builds_shadow_gate(self) -> None:
        memory = research_line.build_search_memory(
            [
                {
                    "candidate_id": "cand_rank",
                    "parent_candidate_id": "cand_bpr_margin_loss",
                    "status": "success",
                    "decision": "keep",
                    "producer_id": "llm_algorithm",
                    "producer_role": "mechanism_discovery",
                    "route_decision": {"score": 0.42, "reason": "family=cand_bpr_margin_loss; action=rank_aware_loss"},
                    "result": {"ndcg@10": 0.281},
                },
                {
                    "candidate_id": "cand_aux",
                    "parent_candidate_id": "cand_lightgcn_aux_alignment_loss",
                    "status": "success",
                    "decision": "revise",
                    "producer_id": "heuristic_algorithm",
                    "producer_role": "template_expansion",
                    "route_decision": {"score": 0.31, "reason": "family=cand_lightgcn_aux_alignment_loss; action=auxiliary_loss"},
                    "result": {"ndcg@10": 0.275},
                },
                {
                    "candidate_id": "cand_bad",
                    "parent_candidate_id": "cand_bad",
                    "status": "crash",
                    "decision": "crash",
                    "producer_id": "repair_ablation",
                    "route_decision": {"score": -0.15, "reason": "family=cand_bad; action=regularization"},
                },
                {
                    "event": "proposal_rejected",
                    "candidate_id": "proposal_bad",
                    "action_type": "regularization",
                    "producer_id": "repair_ablation",
                    "errors": ["duplicate parameter_signature already run"],
                },
            ]
        )

        update = research_line.meta_research_update(memory)

        self.assertEqual(update["controller_version"], "meta_research.v2")
        self.assertIn("action_stats", memory)
        self.assertIn("rank_aware_loss", memory["action_stats"])
        self.assertGreaterEqual(len(update["meta_update_proposals"]), 4)
        self.assertEqual(update["offline_replay"]["status"], "completed")
        self.assertIn("challenger", update["shadow_evaluation"])
        self.assertIn(update["promotion_gate"]["decision"], {"hold_shadow", "candidate_for_independent_promotion"})

    def test_comparison_planner_writes_pair_commands_inside_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            args = [
                "--old-root",
                str(ROOT),
                "--comparison-root",
                str(Path(tmp) / "runs"),
                "--out",
                str(out),
                "--rounds",
                "2",
                "--repeats",
                "2",
            ]

            exit_code = comparison.main(args)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload["pairs"]), 2)
        self.assertIn("research_line", payload["pairs"][0])
        self.assertIn("original", payload["pairs"][0])


if __name__ == "__main__":
    unittest.main()
