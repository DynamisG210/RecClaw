"""Common-immutable campaign compilation memoization.

The mechanism-space compiler remains the sole authority.  This layer only
memoizes its frozen report by the exact canonical program bytes so repeated
validation cannot become a treatment-, Arm-, order-, or round-dependent cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from recclaw_core.mechanism_space import (
    CompileReportV1,
    compile_program,
    compile_program_bytes,
)
from recclaw_core.mechanism_space.canonical import (
    StrictJsonError,
    canonical_bytes,
    snapshot_json,
)


@lru_cache(maxsize=2048)
def _compile_canonical_program_bytes(
    canonical_program_bytes: bytes,
) -> CompileReportV1:
    return compile_program_bytes(canonical_program_bytes)


def compile_campaign_program(
    program: Mapping[str, Any],
) -> CompileReportV1:
    """Return the exact compiler report for canonical program content.

    Invalid caller objects retain the compiler's original error behavior and
    are deliberately not cached.  Valid JSON mappings are keyed by all exact
    canonical bytes; the cached value is a frozen ``CompileReportV1``.
    """

    try:
        raw = canonical_bytes(snapshot_json(dict(program)))
    except (StrictJsonError, TypeError, ValueError):
        return compile_program(program)
    return _compile_canonical_program_bytes(raw)


def compilation_cache_projection() -> dict[str, int | str]:
    info = _compile_canonical_program_bytes.cache_info()
    return {
        "scope": "COMMON_IMMUTABLE",
        "identity": "EXACT_CANONICAL_PROGRAM_BYTES",
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": int(info.maxsize or 0),
        "currsize": info.currsize,
    }


__all__ = (
    "compile_campaign_program",
    "compilation_cache_projection",
)
