#!/usr/bin/env python3
"""Read-only verifier for a completed M6R fixed training canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    EXPECTED_TABLES,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    training_runtime_release_digest,
)


def verify(canary_root: Path) -> dict[str, object]:
    result = json.loads(
        (canary_root / "M6R_FIXED_TRAINING_CANARY_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    db_path = canary_root / "neutral" / "experiment.sqlite3"
    artifact_root = canary_root / "neutral" / "artifacts"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        claims = [dict(row) for row in connection.execute("SELECT * FROM execution_claims")]
        ledger = {
            str(dimension): int(quantity)
            for dimension, quantity in connection.execute(
                """
                SELECT dimension, SUM(quantity)
                FROM resource_ledger
                GROUP BY dimension
                """
            )
        }
        artifacts = [dict(row) for row in connection.execute("SELECT * FROM artifact_index")]
    finally:
        connection.close()

    artifact_failures = []
    for row in artifacts:
        path = artifact_root / str(row["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or bytes_sha256(path.read_bytes()) != row["sha256"]
        ):
            artifact_failures.append(str(row["artifact_id"]))
    required_artifact_types = {
        "COMMON_RESULT_CLOSURE_V2",
        "EXECUTION_START_CONFIRMATION_V1",
        "EXECUTION_START_RECEIPT_V2",
        "RAW_RESULT_ENVELOPE_V2",
        "TRAINING_RAW_RUN_OUTPUT_V1",
        "TRAINING_RESOURCE_ACCOUNTING_V1",
    }
    observed_types = {str(row["artifact_type"]) for row in artifacts}
    file_projection = [
        {
            "path": path.relative_to(canary_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in canary_root.rglob("*") if item.is_file())
    ]
    artifact_tree_digest = sha256_digest(file_projection)
    claim = claims[0] if len(claims) == 1 else {}
    gates = {
        "artifact_index_exact_bytes": not artifact_failures,
        "claim_exactly_once": len(claims) == 1,
        "claim_release_closed": (
            claim.get("claim_state") == "FINISHED"
            and claim.get("attempt_state") == "START_CONFIRMED"
            and claim.get("execution_debited") == 1
            and claim.get("runtime_release_digest")
            == result["runtime_release_digest"]
            and claim.get("runtime_binding_digest")
            == result["runtime_binding_digest"]
        ),
        "current_release_identity": (
            training_runtime_release_digest()
            == result["runtime_release_digest"]
        ),
        "exact_eight_tables": tables == EXPECTED_TABLES,
        "ledger_exact": (
            ledger.get("ORDINARY_EXECUTION") == 1
            and ledger.get("GPU_DEVICE_TIME_MS", -1) >= 0
            and ledger.get("GPU_COST_MICROUNITS", -1) >= 0
            and ledger.get("WALL_TIME_MS")
            == ledger.get("GPU_DEVICE_TIME_MS")
        ),
        "required_artifacts": required_artifact_types.issubset(observed_types),
        "result_verdict": result["verdict"] == "PASS",
        "sqlite_integrity": integrity == "ok" and not foreign_keys,
    }
    return {
        "artifact_count": len(artifacts),
        "artifact_failures": artifact_failures,
        "artifact_tree_digest": artifact_tree_digest,
        "artifact_tree_projection": "SORTED_RELATIVE_PATH_AND_SHA256_V1",
        "gates": gates,
        "runtime_release_digest": result["runtime_release_digest"],
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("canary_root", type=Path)
    args = parser.parse_args()
    report = verify(args.canary_root.resolve())
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
