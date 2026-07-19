"""Create an arm-blinded candidate-confirmation packet from one AB-002 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from recclaw_phase1.ab002_launcher import canonical_write, sha256_file


BLIND_ID = re.compile(r"^AB002-BLIND-[A-Z0-9]{4,32}$")
SOURCE_PATHS = (
    "configs",
    "recclaw_ext",
    "scripts/run_candidate.py",
    "scripts/collect_result.py",
)
PILOT_PATHS = (
    "candidate_proposals.jsonl",
    "results.csv",
    "candidates",
    "overrides",
)


def _tree_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def create_blind_packet(
    *, run_root: Path, blind_root: Path, blind_id: str, mapping_output: Path
) -> dict[str, Any]:
    if not BLIND_ID.fullmatch(blind_id):
        raise ValueError("blind id must match AB002-BLIND-[A-Z0-9]{4,32}")
    source_run = run_root.expanduser().resolve()
    output = blind_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"blind packet root must be absent: {output}")
    if any(component.lower() in {"control", "treatment"} for component in output.parts):
        raise ValueError("blind packet path contains an arm-label component")
    source = source_run / "source"
    pilot = source_run / "outputs" / "pilot"
    if not source.is_dir() or not pilot.is_dir():
        raise ValueError("run root lacks materialized source or pilot output")
    (output / "source").mkdir(parents=True)
    (output / "pilot").mkdir(parents=True)
    for relative in SOURCE_PATHS:
        origin = source / relative
        target = output / "source" / relative
        if not origin.exists():
            raise ValueError(f"blind source input missing: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target)
        else:
            shutil.copy2(origin, target)
    for relative in PILOT_PATHS:
        origin = pilot / relative
        target = output / "pilot" / relative
        if not origin.exists():
            raise ValueError(f"blind pilot input missing: {origin}")
        if origin.is_dir():
            shutil.copytree(origin, target)
        else:
            shutil.copy2(origin, target)
    ledger = source_run / "candidate_execution_budget.jsonl"
    if not ledger.is_file():
        raise ValueError("candidate execution ledger missing")
    shutil.copy2(ledger, output / "candidate_execution_budget.jsonl")
    rows = _tree_rows(output)
    tree_digest = hashlib.sha256()
    for row in rows:
        tree_digest.update(row["path"].encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(row["sha256"].encode("ascii"))
        tree_digest.update(b"\0")
    record = {
        "record_type": "RECCLAW_PHASE1_AB002_BLINDED_RUN_PACKET",
        "schema_version": "recclaw.phase1.ab002.blinded_run_packet.v1",
        "status": "LOCAL_COMPLETE",
        "blind_id": blind_id,
        "packet_root": str(output),
        "file_count": len(rows),
        "file_rows": rows,
        "tree_digest_over_file_digests": tree_digest.hexdigest(),
        "arm_label_present": False,
        "guard_classification_included": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    canonical_write(output / "blind_packet_manifest.json", record)
    mapping = {
        "record_type": "RECCLAW_PHASE1_AB002_BLIND_MAPPING",
        "schema_version": "recclaw.phase1.ab002.blind_mapping.v1",
        "status": "LOCAL_COMPLETE",
        "blind_id": blind_id,
        "source_run_root": str(source_run),
        "blind_packet_root": str(output),
        "blind_packet_manifest_sha256": sha256_file(output / "blind_packet_manifest.json"),
        "not_visible_to_neutral_scoring": True,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    canonical_write(mapping_output.expanduser().resolve(), mapping)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one arm-blinded AB-002 run packet")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--blind-id", required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = create_blind_packet(
        run_root=args.run_root,
        blind_root=args.blind_root,
        blind_id=args.blind_id,
        mapping_output=args.mapping_output,
    )
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
