"""Canonical lineage identity projected from compiler-owned proposal facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def exact_parent_candidate_id(proposal: Any) -> str | None:
    """Return one exact parent without treating descriptive text as identity."""

    top_level = getattr(proposal, "parent_candidate_id", None)
    spec_parent = getattr(getattr(proposal, "spec", None), "parent_candidate_id", None)
    for field_name, value in (
        ("proposal.parent_candidate_id", top_level),
        ("proposal.spec.parent_candidate_id", spec_parent),
    ):
        if value is not None and (
            not isinstance(value, str) or not value or value != value.strip()
        ):
            raise ValueError(f"{field_name} is not normalized")
    if top_level is not None and spec_parent is not None and top_level != spec_parent:
        raise ValueError("proposal parent identity conflicts with spec parent identity")
    direct = top_level if top_level is not None else spec_parent

    program = getattr(proposal, "mechanism_program", None)
    payload = (
        program.get("program_payload", program)
        if isinstance(program, Mapping)
        else {}
    )
    parent_refs = payload.get("parent_refs", ()) if isinstance(payload, Mapping) else ()
    compiler_parents: set[str] = set()
    if isinstance(parent_refs, Sequence) and not isinstance(parent_refs, (str, bytes)):
        for ref in parent_refs:
            candidate_id = ref.get("candidate_id") if isinstance(ref, Mapping) else None
            if isinstance(candidate_id, str) and candidate_id and candidate_id == candidate_id.strip():
                compiler_parents.add(candidate_id)

    if direct is not None and compiler_parents and direct not in compiler_parents:
        raise ValueError("proposal parent identity conflicts with compiler parent_refs")
    if len(compiler_parents) == 1:
        return next(iter(compiler_parents))
    return direct
