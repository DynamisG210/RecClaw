from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_reflection_pilot as pilot  # noqa: E402


class ReflectionPilotTests(unittest.TestCase):
    def test_choose_proposal_source_defaults_to_heuristic_without_key(self) -> None:
        self.assertEqual(pilot.choose_proposal_source(None, env={}), "heuristic")
        self.assertEqual(pilot.choose_proposal_source("llm", env={}), "llm")

    def test_choose_loop_mode_matches_proposal_source_by_default(self) -> None:
        self.assertEqual(pilot.choose_loop_mode(None, "heuristic"), "mixed")
        self.assertEqual(pilot.choose_loop_mode(None, "llm"), "auto")
        self.assertEqual(pilot.choose_loop_mode("explore", "heuristic"), "explore")

    def test_choose_llm_provider_matches_available_key(self) -> None:
        self.assertEqual(pilot.choose_llm_provider(None, env={"DEEPSEEK_API_KEY": "x"}), "deepseek")
        self.assertEqual(pilot.choose_llm_provider(None, env={"OPENAI_API_KEY": "x"}), "openai")
        self.assertEqual(pilot.choose_llm_provider("compatible", env={}), "compatible")
        self.assertEqual(pilot.default_llm_key_env("deepseek"), "DEEPSEEK_API_KEY")
        self.assertEqual(pilot.default_llm_key_env("openai"), "OPENAI_API_KEY")

    def test_validate_llm_env_requires_matching_key_for_llm(self) -> None:
        pilot.validate_llm_env(proposal_source="heuristic", key_env="DEEPSEEK_API_KEY", env={})
        with self.assertRaisesRegex(ValueError, "requires DEEPSEEK_API_KEY"):
            pilot.validate_llm_env(proposal_source="llm", key_env="DEEPSEEK_API_KEY", env={})
        pilot.validate_llm_env(proposal_source="llm", key_env="DEEPSEEK_API_KEY", env={"DEEPSEEK_API_KEY": "x"})

    def test_resolve_baseline_dir_prefers_explicit_then_env(self) -> None:
        explicit = pilot.resolve_baseline_dir("/tmp/explicit", env={"RECCLAW_BASELINE_DIR": "/tmp/env"})
        from_env = pilot.resolve_baseline_dir(None, env={"RECCLAW_BASELINE_DIR": "/tmp/env"})
        self.assertEqual(explicit, Path("/tmp/explicit"))
        self.assertEqual(from_env, Path("/tmp/env"))

    def test_validate_agent_baseline_dir_requires_top_level_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "top-level .log"):
                pilot.validate_agent_baseline_dir(path)
            (path / "baseline_lightgcn.log").write_text("test result: {'ndcg@10': 0.1}\n")
            pilot.validate_agent_baseline_dir(path)

    def test_dry_run_prints_complete_commands_without_creating_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--search-seed",
                        "42",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit",
                        "--rounds",
                        "50",
                        "--gpu-id",
                        "2",
                        "--proposal-source",
                        "heuristic",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn("lint", payload["commands"])
            self.assertIn("initial_tree", payload["commands"])
            self.assertIn("initial_summary", payload["commands"])
            self.assertIn("agent", payload["commands"])
            self.assertIn("--rounds 50", payload["commands"]["agent"])
            self.assertIn("--start-round 1", payload["commands"]["agent"])
            self.assertIn("--loop-mode mixed", payload["commands"]["agent"])
            self.assertIn("--refresh-experience-every 10", payload["commands"]["agent"])
            self.assertIn("--set gpu_id=2", payload["commands"]["agent"])
            self.assertIn("--baseline-dir", payload["commands"]["agent"])
            self.assertIn("--search-intensity algorithm_first", payload["commands"]["agent"])
            self.assertIn("--algorithm-budget-per-window 3", payload["commands"]["agent"])
            self.assertIn("--proposal-count 6", payload["commands"]["agent"])
            self.assertNotIn("--allow-llm-fallback", payload["commands"]["agent"])
            self.assertEqual(payload["loop_mode"], "mixed")
            self.assertEqual(payload["search_intensity"], "algorithm_first")
            self.assertEqual(payload["llm_provider"], "deepseek")
            self.assertEqual(payload["llm_api_key_env"], "DEEPSEEK_API_KEY")
            self.assertTrue(payload["baseline_dir"].endswith("results/baseline"))
            self.assertFalse((Path(tmp) / "unit").exists())

    def test_llm_dry_run_defaults_to_auto_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--search-seed",
                        "42",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit-llm",
                        "--proposal-source",
                        "llm",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["proposal_source"], "llm")
            self.assertEqual(payload["loop_mode"], "auto")
            self.assertEqual(payload["llm_provider"], "deepseek")
            self.assertEqual(payload["llm_api_key_env"], "DEEPSEEK_API_KEY")
            self.assertIn("--loop-mode auto", payload["commands"]["agent"])
            self.assertIn("--llm-provider deepseek", payload["commands"]["agent"])
            self.assertIn("--llm-api-key-env DEEPSEEK_API_KEY", payload["commands"]["agent"])
            self.assertIn("--search-intensity algorithm_first", payload["commands"]["agent"])
            self.assertIn("--max-pending-implemented 6", payload["commands"]["agent"])
            self.assertIn("--max-implement-per-round 2", payload["commands"]["agent"])
            self.assertIn("--seed-validation-min-metric 0.274", payload["commands"]["agent"])
            self.assertNotIn("--allow-llm-fallback", payload["commands"]["agent"])

    def test_llm_dry_run_can_opt_into_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--search-seed",
                        "42",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit-llm-fallback",
                        "--proposal-source",
                        "llm",
                        "--allow-llm-fallback",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn("--allow-llm-fallback", payload["commands"]["agent"])

    def test_llm_dry_run_can_select_openai_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--search-seed",
                        "42",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit-openai",
                        "--proposal-source",
                        "llm",
                        "--llm-provider",
                        "openai",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["llm_provider"], "openai")
            self.assertEqual(payload["llm_api_key_env"], "OPENAI_API_KEY")
            self.assertIn("--llm-provider openai", payload["commands"]["agent"])
            self.assertIn("--llm-api-key-env OPENAI_API_KEY", payload["commands"]["agent"])

    def test_continuation_dry_run_passes_start_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--search-seed",
                        "42",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit-continue",
                        "--proposal-source",
                        "heuristic",
                        "--start-round",
                        "51",
                        "--rounds",
                        "200",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["start_round"], 51)
            self.assertIn("--start-round 51", payload["commands"]["agent"])
            self.assertIn("--rounds 200", payload["commands"]["agent"])
            self.assertFalse((Path(tmp) / "unit-continue").exists())

    def test_seed_runtime_artifacts_copies_known_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed"
            seed.mkdir()
            (seed / "agent_memory.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
            (seed / "results.csv").write_text("candidate_id,ndcg@10\ncand,0.1\n", encoding="utf-8")
            (seed / "ignore.tmp").write_text("ignore", encoding="utf-8")
            paths = pilot.build_paths(root, "target")
            pilot.ensure_runtime_dirs(paths)

            copied = pilot.seed_runtime_artifacts(paths, seed)

            self.assertEqual(copied, ["agent_memory.jsonl", "results.csv"])
            self.assertTrue(paths.memory.exists())
            self.assertTrue(paths.results_csv.exists())
            self.assertFalse((paths.run_dir / "ignore.tmp").exists())


if __name__ == "__main__":
    unittest.main()
