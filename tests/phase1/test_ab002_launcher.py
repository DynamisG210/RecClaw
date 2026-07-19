from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_launcher import (
    FROZEN_GUARD_FILES,
    build_launch_plan,
    materialize_arm,
    validate_runtime_root,
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

    def test_dry_plan_is_external_and_has_no_seed_from(self) -> None:
        before = _tree_digest(PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_launch_plan(
                runtime_root=Path(temporary) / "runtime",
                arm="control",
                search_seed=44,
                gpu_id=0,
                pair_id="PAIR-44",
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

    def test_s0_manifest_has_no_historical_outputs(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/s0_manifest.json").read_text()
        )
        self.assertEqual(0, manifest["historical_search_output_count"])
        self.assertEqual([], manifest["seeded_files"])


if __name__ == "__main__":
    unittest.main()
