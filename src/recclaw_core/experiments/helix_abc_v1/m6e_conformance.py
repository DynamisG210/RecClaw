"""Prospective gate for using the M6E training substrate in a fresh Pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import sha256_digest
from .training_runtime_release import training_runtime_release


M6E_CONFORMANCE_PACKET = (
    Path("docs")
    / "research_line"
    / "m6e"
    / "M6E_TRAINING_RUNTIME_CONFORMANCE_PACKET.json"
)
REQUIRED_HANDLER_FAMILIES = frozenset({"BPR", "LightGCN", "NGCF", "SGL"})


def require_m6e_conformance_packet(project_root: Path) -> dict[str, Any]:
    path = project_root.resolve() / M6E_CONFORMANCE_PACKET
    if not path.is_file():
        raise RuntimeError("M6E conformance packet is absent")
    document = json.loads(path.read_text(encoding="utf-8"))
    claimed_digest = str(document.pop("content_digest", ""))
    if claimed_digest != sha256_digest(document):
        raise RuntimeError("M6E conformance packet digest mismatch")
    if (
        document.get("verdict") != "PASS"
        or int(document.get("P0", -1)) != 0
        or int(document.get("P1", -1)) != 0
    ):
        raise RuntimeError("M6E conformance packet is not independently clear")
    if (
        document.get("training_runtime_release_digest")
        != training_runtime_release().digest
    ):
        raise RuntimeError("M6E packet does not bind the active training release")
    canaries = tuple(document.get("fixed_training_canaries", ()))
    passed_families = {
        str(row["model"])
        for row in canaries
        if row.get("verdict") == "PASS"
        and row.get("runtime_release_digest")
        == training_runtime_release().digest
    }
    if passed_families != REQUIRED_HANDLER_FAMILIES:
        raise RuntimeError("M6E handler-family conformance coverage is incomplete")
    failure = dict(document.get("forced_failure_rehearsal", {}))
    if (
        failure.get("verdict") != "PASS"
        or failure.get("classified_outcome") != "RUNTIME_FAILURE"
        or failure.get("successful_training_count") != 0
    ):
        raise RuntimeError("M6E forced-failure classification is not closed")
    audit = dict(document.get("full_authoritative_audit_rehearsal", {}))
    if audit.get("verdict") != "PASS" or not audit.get("barriers_closed"):
        raise RuntimeError("M6E authoritative Pilot audit rehearsal did not pass")
    return {**document, "content_digest": claimed_digest}


__all__ = [
    "M6E_CONFORMANCE_PACKET",
    "REQUIRED_HANDLER_FAMILIES",
    "require_m6e_conformance_packet",
]
