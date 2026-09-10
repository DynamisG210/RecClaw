#!/usr/bin/env python3
"""Reversibly compact sealed ResearchCampaign trace mirrors.

The checkpoint remains the authoritative complete CampaignRoundRecord.  This
tool only touches rounds that the durable campaign state has already passed.
The exact original trace bytes are retained in a verified deterministic gzip.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_json_bytes
from recclaw_core.research_line.campaign import (
    CampaignRoundRecord,
    CampaignState,
    compact_campaign_round_trace,
)


_TRACE_RE = re.compile(r"^ROUND_(\d+)_TRACE\.json$")
_TERMINAL_STATUSES = frozenset(
    {
        "OUTCOME_MISSING",
        "TYPED_FAILURE",
        "TYPED_EPISODE",
        "TYPED_FAILURE_NO_METRIC",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _gzip_exact(path: Path, payload: bytes) -> tuple[str, int]:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(payload)
        raw.flush()
        os.fsync(raw.fileno())
    compressed = temporary.read_bytes()
    with gzip.open(temporary, "rb") as handle:
        restored = handle.read()
    if restored != payload:
        raise RuntimeError(f"gzip round-trip mismatch for {path.name}")
    os.replace(temporary, path)
    return _sha256(compressed), len(compressed)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _selected_rounds(root: Path, requested: set[int]) -> tuple[tuple[int, Path], ...]:
    rows: list[tuple[int, Path]] = []
    for path in sorted(root.glob("ROUND_*_TRACE.json")):
        match = _TRACE_RE.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if not requested or index in requested:
            rows.append((index, path))
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--round-index", type=int, action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    root = args.run_root.resolve()
    state = _load_pickle(root / "CAMPAIGN_STATE.pkl")
    if not isinstance(state, CampaignState):
        raise RuntimeError("CAMPAIGN_STATE.pkl is not a CampaignState")
    archive_root = root / "ROUND_TRACE_ARCHIVE_GZIP_V1"

    report: list[dict[str, Any]] = []
    for index, trace_path in _selected_rounds(root, set(args.round_index)):
        checkpoint_path = root / f"ROUND_{index:02d}_CHECKPOINT.pkl"
        if not checkpoint_path.is_file():
            report.append({"round_index": index, "status": "SKIPPED_NO_CHECKPOINT"})
            continue
        checkpoint_payload = checkpoint_path.read_bytes()
        record = pickle.loads(checkpoint_payload)
        if not isinstance(record, CampaignRoundRecord) or record.round_index != index:
            raise RuntimeError(f"round {index} checkpoint identity drift")
        if record.status not in _TERMINAL_STATUSES or state.next_round_index <= index:
            report.append({"round_index": index, "status": "SKIPPED_NOT_SEALED"})
            continue

        trace_payload = trace_path.read_bytes()
        trace = json.loads(trace_payload)
        if trace.get("record_digest") != record.digest:
            raise RuntimeError(f"round {index} trace/checkpoint digest drift")
        checkpoint_sha = _sha256(checkpoint_payload)
        if trace.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"round {index} checkpoint SHA drift")

        if args.restore:
            if trace.get("schema") != "recclaw.research-line.campaign-round-trace.v2":
                report.append({"round_index": index, "status": "SKIPPED_NO_ARCHIVE"})
                continue
            archived = trace.get("archived_trace")
            if not isinstance(archived, dict):
                report.append({"round_index": index, "status": "SKIPPED_NO_ARCHIVE"})
                continue
            gzip_ref = str(archived["gzip_ref"])
            gzip_path = (root / gzip_ref).resolve()
            if gzip_path.parent != archive_root.resolve() or gzip_path.name != Path(
                gzip_ref
            ).name:
                raise RuntimeError(f"round {index} archived trace path drift")
            gzip_payload = gzip_path.read_bytes()
            if (
                len(gzip_payload) != archived["gzip_size"]
                or _sha256(gzip_payload) != archived["gzip_sha256"]
            ):
                raise RuntimeError(f"round {index} archived gzip drift")
            with gzip.open(gzip_path, "rb") as handle:
                original = handle.read()
            if len(original) != archived["original_size"] or _sha256(original) != archived[
                "original_sha256"
            ]:
                raise RuntimeError(f"round {index} archived trace drift")
            original_trace = json.loads(original)
            if (
                original_trace.get("schema")
                != "recclaw.research-line.campaign-round-trace.v1"
                or original_trace.get("record_digest") != record.digest
                or original_trace.get("checkpoint_sha256") != checkpoint_sha
            ):
                raise RuntimeError(f"round {index} archived v1 identity drift")
            _write_atomic(trace_path, original)
            report.append({"round_index": index, "status": "RESTORED"})
            continue
        if trace.get("schema") == "recclaw.research-line.campaign-round-trace.v2":
            report.append({"round_index": index, "status": "ALREADY_COMPACT"})
            continue
        if trace.get("schema") != "recclaw.research-line.campaign-round-trace.v1":
            raise RuntimeError(f"round {index} trace schema drift")

        compact = compact_campaign_round_trace(
            record,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_ref=checkpoint_path.name,
        )
        original_sha = _sha256(trace_payload)
        gzip_name = f"ROUND_{index:02d}_TRACE.{original_sha}.{len(trace_payload)}.json.gz"
        gzip_path = archive_root / gzip_name
        compact_payload = canonical_json_bytes(compact) + b"\n"
        if not args.apply:
            report.append(
                {
                    "round_index": index,
                    "status": "ELIGIBLE",
                    "original_size": len(trace_payload),
                    "compact_size": len(compact_payload),
                }
            )
            continue
        archive_root.mkdir(mode=0o755, exist_ok=True)
        if gzip_path.exists():
            with gzip.open(gzip_path, "rb") as handle:
                if handle.read() != trace_payload:
                    raise RuntimeError(f"round {index} existing archive drift")
            gzip_payload = gzip_path.read_bytes()
            gzip_sha, gzip_size = _sha256(gzip_payload), len(gzip_payload)
        else:
            gzip_sha, gzip_size = _gzip_exact(gzip_path, trace_payload)
        compact = {
            **compact,
            "archived_trace": {
                "gzip_ref": str(gzip_path.relative_to(root)),
                "gzip_sha256": gzip_sha,
                "gzip_size": gzip_size,
                "original_sha256": original_sha,
                "original_size": len(trace_payload),
            },
        }
        compact_payload = canonical_json_bytes(compact) + b"\n"
        _write_atomic(trace_path, compact_payload)
        report.append(
            {
                "round_index": index,
                "status": "COMPACTED",
                "original_size": len(trace_payload),
                "gzip_size": gzip_size,
                "compact_size": len(compact_payload),
            }
        )

    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
