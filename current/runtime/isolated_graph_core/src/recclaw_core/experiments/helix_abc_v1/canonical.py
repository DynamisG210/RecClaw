"""Canonical identity helpers for the M0 contract kernel."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping


class CanonicalizationError(ValueError):
    """Raised when a value cannot enter a canonical identity preimage."""


def canonical_value(value: Any) -> Any:
    """Return a JSON-safe, deterministically ordered value."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite numbers are forbidden")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise CanonicalizationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a canonical value with stable UTF-8 JSON bytes."""

    return json.dumps(
        canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_digest(value: Any) -> str:
    """Return a lowercase SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_id(namespace: str, payload: Any) -> str:
    if not namespace or any(char.isspace() for char in namespace):
        raise CanonicalizationError("identity namespace must be non-empty and whitespace-free")
    return f"{namespace}:{sha256_digest(payload)}"


def validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return normalized


def validate_relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("artifact path must be a normalized root-relative POSIX path")
    normalized = path.as_posix()
    if normalized != value or normalized.startswith("/"):
        raise ValueError("artifact path must already be canonical")
    return normalized

