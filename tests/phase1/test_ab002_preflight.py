from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_preflight import exact_file_identities, tree_identity


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


if __name__ == "__main__":
    unittest.main()
