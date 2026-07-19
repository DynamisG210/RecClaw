from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recclaw_phase1.ab002_launcher import (
    CANARY_SEED,
    FROZEN_GUARD_FILES,
    FULL_SEARCH_SEEDS,
    _build_execution_environment,
    apply_frozen_treatment_overlay,
    build_launch_plan,
    expected_candidate_executions,
    expected_gpu_id,
    expected_pair_id,
    materialize_arm,
    validate_runtime_root,
)
from recclaw_phase1.ab002_s0 import (
    FORBIDDEN_RUNTIME_TOKENS,
    INITIAL_STATE_MANIFEST,
    LEAKAGE_AUDIT,
    S0_ID,
    validate_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class AB002LauncherTests(unittest.TestCase):
    def test_runtime_root_rejects_source_and_parent(self) -> None:
        with self.assertRaises(ValueError):
            validate_runtime_root(PROJECT_ROOT / "runtime")
        with self.assertRaises(ValueError):
            validate_runtime_root(PROJECT_ROOT.parent)

    def test_control_materialization_contains_no_guard_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = materialize_arm(
                runtime_root=Path(temporary) / "runtime",
                arm="control",
                search_seed=42,
            )
            source = Path(record["source_root"])
            self.assertEqual([], record["integration_files"])
            self.assertFalse((source / "src").exists())
            self.assertFalse(any("evidence_guard" in path.name for path in source.rglob("*")))
            self.assertNotIn(
                b"recclaw_core.exploration",
                (source / "scripts/agent.py").read_bytes(),
            )

    def test_treatment_materialization_uses_only_frozen_guard_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = materialize_arm(
                runtime_root=Path(temporary) / "runtime",
                arm="treatment",
                search_seed=43,
            )
            integration = Path(record["integration_root"])
            actual = {
                row["path"]: row["sha256"]
                for row in record["integration_files"]
                if row["path"].startswith("src/")
            }
            self.assertEqual(FROZEN_GUARD_FILES, actual)
            self.assertIn(
                b"OriginalRecClawGuardHook",
                Path(record["source_root"], "scripts/agent.py").read_bytes(),
            )
            self.assertTrue((integration / "evidence_guard_contract.json").is_file())
            runtime_contract = (integration / "evidence_guard_contract.json").read_text()
            for token in ("0.2908", "0.289067", "0.274367", "historical_v4m"):
                self.assertNotIn(token, runtime_contract)

    def test_two_arms_share_exact_s0_except_frozen_treatment_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            control = materialize_arm(runtime_root=runtime, arm="control", search_seed=42)
            treatment = materialize_arm(runtime_root=runtime, arm="treatment", search_seed=42)
            self.assertEqual(S0_ID, control["s0_id"])
            self.assertEqual(control["s0_source_tree_sha256"], treatment["s0_source_tree_sha256"])
            control_rows = {row["path"]: row for row in control["source_files"]}
            treatment_rows = {row["path"]: row for row in treatment["source_files"]}
            self.assertEqual(set(control_rows), set(treatment_rows))
            for path in control_rows:
                if path == "scripts/agent.py":
                    self.assertEqual(
                        control_rows[path]["sha256"],
                        treatment_rows[path]["common_s0_sha256"],
                    )
                    self.assertNotEqual(control_rows[path]["sha256"], treatment_rows[path]["sha256"])
                else:
                    self.assertEqual(control_rows[path]["sha256"], treatment_rows[path]["sha256"])

    def test_treatment_materialization_does_not_require_git(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "recclaw_phase1.ab002_launcher.subprocess.run",
                side_effect=AssertionError("materialization must not invoke an external process"),
            ):
                record = materialize_arm(
                    runtime_root=Path(temporary) / "runtime",
                    arm="treatment",
                    search_seed=42,
                )
        self.assertEqual("LOCAL_COMPLETE", record["status"])

    def test_frozen_overlay_rejects_single_context_fault(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "agent.py"
            overlay = Path(temporary) / "overlay.patch"
            source.write_bytes((PROJECT_ROOT / "scripts/agent.py").read_bytes())
            overlay_bytes = (PROJECT_ROOT / "phase1/overlays/treatment_agent.patch").read_bytes()
            overlay.write_bytes(overlay_bytes.replace(b" try:\n", b" nope:\n", 1))
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                apply_frozen_treatment_overlay(source, overlay)

    def test_control_execution_environment_clears_ambient_guard_paths(self) -> None:
        control_plan = {
            "environment_without_secrets": {
                "RECCLAW_AB_ARM": "control",
                "RECBOLE_ROOT": "/frozen/recbole",
                "PYTHONNOUSERSITE": "1",
                "OPENAI_API_KEY": "FROM_RECCLAW_BROKER_CLIENT_TOKEN",
            }
        }
        treatment_plan = {
            "environment_without_secrets": {
                **control_plan["environment_without_secrets"],
                "RECCLAW_AB_ARM": "treatment",
                "PYTHONPATH": "/frozen/treatment/integration",
                "RECCLAW_EVIDENCE_GUARD_MODE": "active",
                "RECCLAW_EVIDENCE_GUARD_CONTRACT": "/frozen/treatment/contract",
                "RECCLAW_EVIDENCE_GUARD_FEEDBACK": "/frozen/treatment/feedback",
            }
        }
        ambient = {
            "PYTHONPATH": "/ambient/guard",
            "RECCLAW_EVIDENCE_GUARD_MODE": "ambient",
            "RECCLAW_EVIDENCE_GUARD_CONTRACT": "/ambient/contract",
            "RECCLAW_EVIDENCE_GUARD_FEEDBACK": "/ambient/feedback",
            "RECCLAW_LAB_LLM_API_KEY": "raw-upstream-secret",
            "RECCLAW_LAB_LLM_BASE_URL": "https://upstream.example.invalid/v1",
            "RECCLAW_BROKER_CLIENT_TOKEN": "ambient-broker-token",
        }
        with mock.patch.dict(os.environ, ambient, clear=False):
            control = _build_execution_environment(control_plan, client_token="token")
            treatment = _build_execution_environment(treatment_plan, client_token="token")
        for key in ambient:
            self.assertNotIn(key, control)
        for key in (
            "RECCLAW_LAB_LLM_API_KEY",
            "RECCLAW_LAB_LLM_BASE_URL",
            "RECCLAW_BROKER_CLIENT_TOKEN",
        ):
            self.assertNotIn(key, treatment)
        self.assertEqual("token", control["OPENAI_API_KEY"])
        self.assertEqual("token", treatment["OPENAI_API_KEY"])
        self.assertEqual("/frozen/treatment/integration", treatment["PYTHONPATH"])
        self.assertEqual("active", treatment["RECCLAW_EVIDENCE_GUARD_MODE"])
        self.assertEqual(
            "/frozen/treatment/contract",
            treatment["RECCLAW_EVIDENCE_GUARD_CONTRACT"],
        )
        self.assertEqual(
            "/frozen/treatment/feedback",
            treatment["RECCLAW_EVIDENCE_GUARD_FEEDBACK"],
        )
        self.assertEqual("/frozen/recbole", control["RECBOLE_ROOT"])
        self.assertEqual("1", control["PYTHONNOUSERSITE"])

    def test_dry_plan_is_external_and_has_no_seed_from(self) -> None:
        before = _tree_digest(PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_launch_plan(
                runtime_root=Path(temporary) / "runtime",
                arm="control",
                search_seed=44,
                gpu_id=0,
                pair_id="AB002-SEED-44",
                broker_url="http://127.0.0.1:18080",
                baseline_dir=Path(temporary) / "baseline",
                python_executable="python3",
            )
        after = _tree_digest(PROJECT_ROOT)
        self.assertEqual(before, after)
        self.assertFalse(plan["seed_from_argument_present"])
        self.assertTrue(plan["all_outputs_external"])
        self.assertNotIn("--seed-from", plan["command_argv"])
        self.assertNotIn("PYTHONPATH", plan["environment_without_secrets"])
        self.assertEqual(
            "/home/tingrangan/projects/RecBole",
            plan["environment_without_secrets"]["RECBOLE_ROOT"],
        )
        self.assertEqual(20, plan["round_budget"])
        self.assertIn("--llm-temperature", plan["command_argv"])

    def test_materialized_canary_harness_accepts_exact_launch_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            baseline = Path(temporary) / "baseline"
            materialization = materialize_arm(
                runtime_root=runtime,
                arm="control",
                search_seed=CANARY_SEED,
            )
            plan = build_launch_plan(
                runtime_root=runtime,
                arm="control",
                search_seed=CANARY_SEED,
                gpu_id=0,
                pair_id="AB002-SEED-9001",
                broker_url="http://127.0.0.1:18080",
                baseline_dir=baseline,
                python_executable=sys.executable,
            )
            completed = subprocess.run(
                [*plan["command_argv"], "--dry-run"],
                cwd=Path(materialization["source_root"]),
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(1, payload["max_implement_per_round"])
        self.assertIn("--max-implement-per-round 1", payload["commands"]["agent"])

    def test_canary_and_full_seed_contract_is_not_user_reconfigurable(self) -> None:
        self.assertEqual("AB002-SEED-9001", expected_pair_id(CANARY_SEED))
        self.assertEqual(0, expected_gpu_id(CANARY_SEED, "control"))
        self.assertEqual(1, expected_gpu_id(CANARY_SEED, "treatment"))
        self.assertEqual(6, expected_candidate_executions(CANARY_SEED))
        self.assertEqual(20, expected_candidate_executions(42))
        self.assertEqual((42, 43, 44, 45, 46, 47), FULL_SEARCH_SEEDS)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "pair id must bind"):
                build_launch_plan(
                    runtime_root=Path(temporary) / "runtime",
                    arm="control",
                    search_seed=42,
                    gpu_id=0,
                    pair_id="AB002-SEED-43",
                    broker_url="http://127.0.0.1:18080",
                    baseline_dir=Path(temporary) / "baseline",
                    python_executable="python3",
                )
            with self.assertRaisesRegex(ValueError, "GPU assignment drift"):
                build_launch_plan(
                    runtime_root=Path(temporary) / "runtime",
                    arm="control",
                    search_seed=43,
                    gpu_id=0,
                    pair_id="AB002-SEED-43",
                    broker_url="http://127.0.0.1:18080",
                    baseline_dir=Path(temporary) / "baseline",
                    python_executable="python3",
                )

    def test_s0_manifest_has_no_historical_outputs(self) -> None:
        manifest = validate_records()
        audit = json.loads(LEAKAGE_AUDIT.read_text())
        self.assertEqual(S0_ID, manifest["s0_id"])
        self.assertEqual("RECONSTRUCTED_CLEAN_START_S0", manifest["classification"])
        self.assertEqual(0, manifest["historical_search_output_count"])
        self.assertEqual(59, manifest["source_file_count"])
        self.assertEqual(23, manifest["candidate_registry_count"])
        self.assertEqual(
            "NO_KNOWN_HISTORICAL_SEARCH_LEAKAGE_DETECTED", audit["verdict"]
        )
        source = PROJECT_ROOT / manifest["source_root"]
        joined = b"\n".join(path.read_bytes().lower() for path in source.rglob("*") if path.is_file())
        for token in FORBIDDEN_RUNTIME_TOKENS:
            self.assertNotIn(token.encode().lower(), joined)
        self.assertEqual(INITIAL_STATE_MANIFEST, PROJECT_ROOT / "phase1/s0/ab002/initial_state_manifest.json")


if __name__ == "__main__":
    unittest.main()
