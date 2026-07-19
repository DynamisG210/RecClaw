from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_preflight import (
    exact_file_identities,
    resolve_release_identity,
    tree_identity,
    verify_clean_tree_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AB002PreflightTests(unittest.TestCase):
    def test_tree_identity_binds_relative_paths_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.txt").write_bytes(b"two")
            (root / "a.txt").write_bytes(b"one")
            observed, rows = tree_identity(root)
        expected = hashlib.sha256(
            b"a.txt\0one\0b.txt\0two\0"
        ).hexdigest()
        self.assertEqual(expected, observed)
        self.assertEqual({"a.txt", "b.txt"}, set(rows))

    def test_exact_file_identity_single_fault_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime.py").write_bytes(b"expected")
            with self.assertRaisesRegex(ValueError, "identity drift"):
                exact_file_identities(root, {"runtime.py": "0" * 64})

    def test_gitless_release_sidecar_binds_archive_and_exact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            root = outer / "delivery"
            root.mkdir()
            source = root / "source.py"
            source.write_bytes(b"frozen\n")
            sums = root / "SOURCE_SHA256SUMS"
            sums.write_text(f"{_sha256(source)}  source.py\n", encoding="utf-8")
            manifest = root / "clean_tree_manifest.json"
            rows = [
                {"path": "SOURCE_SHA256SUMS", "sha256": _sha256(sums)},
                {"path": "source.py", "sha256": _sha256(source)},
            ]
            manifest.write_text(
                json.dumps({"file_count": len(rows), "files": rows}) + "\n",
                encoding="utf-8",
            )
            archive = outer / "release.tar"
            archive.write_bytes(b"exact archive bytes")
            identity = outer / "release_identity.json"
            payload = {
                "record_type": "RECCLAW_PHASE1_AB002_EXTERNAL_RELEASE_IDENTITY",
                "schema_version": "recclaw.phase1.ab002.external_release_identity.v1",
                "status": "LOCAL_COMPLETE",
                "commit": "1" * 40,
                "tree": "2" * 40,
                "tag": "phase1-ab002-pre-canary-r4",
                "source_archive_path": str(archive),
                "source_archive_sha256": _sha256(archive),
                "source_sha256s_sha256": _sha256(sums),
                "clean_tree_manifest_sha256": _sha256(manifest),
                "authority": "NONE",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
            }
            identity.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self.assertEqual(3, verify_clean_tree_manifest(root, manifest))
            resolved = resolve_release_identity(
                root=root,
                expected_tag="phase1-ab002-pre-canary-r4",
                release_identity_path=identity,
                source_sums_path=sums,
                clean_manifest_path=manifest,
            )
            self.assertEqual(
                "EXTERNAL_RELEASE_ARCHIVE_AND_EXACT_FILE_MANIFEST",
                resolved["verification_mode"],
            )
            archive.write_bytes(b"single fault")
            with self.assertRaisesRegex(ValueError, "archive identity drift"):
                resolve_release_identity(
                    root=root,
                    expected_tag="phase1-ab002-pre-canary-r4",
                    release_identity_path=identity,
                    source_sums_path=sums,
                    clean_manifest_path=manifest,
                )


if __name__ == "__main__":
    unittest.main()
