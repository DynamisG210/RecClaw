from __future__ import annotations

import ast
import re
import sys
import unittest
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "recclaw_core" / "experiments" / "helix_abc_v1"
sys.path.insert(0, str(ROOT / "src"))

from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmPolicyV1,
    ExperimentContractV1,
    FusedSearchFeedbackV1,
    ProposalGenerationSessionV1,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    EXPECTED_TABLES,
)


class M0BoundaryTests(unittest.TestCase):
    def test_shared_and_research_modules_do_not_import_evidence_guard(self) -> None:
        forbidden_imports = {
            "recbole",
            "scripts.agent",
            "scripts.research_line",
            "subprocess",
            "urllib",
        }
        m0_modules = {
            "__init__.py",
            "canonical.py",
            "contracts.py",
            "controllers.py",
            "evidence.py",
            "fusion.py",
            "state_store.py",
        }
        for path in sorted(
            item for item in PACKAGE.glob("*.py") if item.name in m0_modules
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
            self.assertFalse(
                any("evidence_guard" in name for name in imports),
                f"forbidden Evidence Guard import in {path.name}: {sorted(imports)}",
            )
            self.assertFalse(
                imports & forbidden_imports,
                f"M1+ or runtime import in {path.name}: {sorted(imports & forbidden_imports)}",
            )

    def test_research_owned_contracts_have_no_evidence_authority_fields(self) -> None:
        forbidden = {
            "claim_ceiling",
            "evidence_admission",
            "protocol_branch",
            "cross_protocol_contamination",
            "accepted_evidence",
            "permission_decision",
        }
        for model in (
            ExperimentContractV1,
            ArmPolicyV1,
            ProposalGenerationSessionV1,
            FusedSearchFeedbackV1,
        ):
            self.assertFalse({item.name for item in fields(model)} & forbidden)

    def test_migration_declares_exactly_the_eight_m0_tables(self) -> None:
        sql = (
            PACKAGE / "migrations" / "001_minimum_sufficient.sql"
        ).read_text(encoding="utf-8")
        declared = {
            match.group(1)
            for match in re.finditer(
                r"CREATE TABLE IF NOT EXISTS\s+([a-z_]+)", sql, flags=re.IGNORECASE
            )
        }
        self.assertEqual(declared, EXPECTED_TABLES)
        self.assertEqual(len(declared), 8)
        self.assertNotIn("publication", sql.lower())
        self.assertNotIn("custody", sql.lower())
        self.assertNotIn("open_claim", sql.lower())

    def test_m0_package_contains_no_post_m1_runtime_files(self) -> None:
        names = {path.name for path in PACKAGE.rglob("*") if path.is_file()}
        for forbidden in (
            "runner.py",
            "broker.py",
            "guard_adapter.py",
            "router.py",
            "meta.py",
            "producer.py",
        ):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()
