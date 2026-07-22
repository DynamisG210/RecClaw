"""Strict JSON snapshots and domain-separated RFC 8785 identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import rfc8785


class StrictJsonError(ValueError):
    """Stable failure raised before an untrusted JSON value is used."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_json_bytes(raw: bytes | bytearray | memoryview) -> Any:
    payload = bytes(raw)
    if payload.startswith(b"\xef\xbb\xbf"):
        raise StrictJsonError("JSON_UTF8_BOM", "UTF-8 JSON must not contain a BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictJsonError("JSON_INVALID_UTF8", "input is not valid UTF-8") from exc

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJsonError("JSON_DUPLICATE_KEY", f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise StrictJsonError("JSON_NONFINITE_NUMBER", f"non-finite number: {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except StrictJsonError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJsonError("JSON_SYNTAX_ERROR", exc.msg) from exc
    _reject_non_json_values(value)
    canonical_bytes(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        result = rfc8785.dumps(value)
    except Exception as exc:
        raise StrictJsonError(
            "JSON_RFC8785_DOMAIN",
            f"value is outside the RFC 8785 domain: {exc}",
        ) from exc
    if not isinstance(result, bytes):  # pragma: no cover - dependency API guard
        raise StrictJsonError("JSON_CANONICALIZER_ERROR", "canonicalizer returned non-bytes")
    return result


def domain_sha256(domain: str, value: Any) -> str:
    if not domain or "\x00" in domain:
        raise ValueError("hash domain must be a non-empty NUL-free string")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + canonical_bytes(value)).hexdigest()


def snapshot_json(value: Any) -> Any:
    """Snapshot a public Python value through the strict canonical boundary."""

    return load_json_bytes(canonical_bytes(value))


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


def _reject_non_json_values(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise StrictJsonError("JSON_LONE_SURROGATE", "JSON strings cannot contain lone surrogates")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJsonError("JSON_NONFINITE_NUMBER", "JSON numbers must be finite")
        return
    if value is None or isinstance(value, int | bool):
        return
    if isinstance(value, list):
        for item in value:
            _reject_non_json_values(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJsonError("JSON_NONSTRING_KEY", "JSON object keys must be strings")
            _reject_non_json_values(key)
            _reject_non_json_values(item)
        return
    raise StrictJsonError("JSON_UNSUPPORTED_TYPE", f"unsupported JSON type: {type(value).__name__}")


__all__ = [
    "StrictJsonError",
    "canonical_bytes",
    "deep_freeze",
    "deep_thaw",
    "domain_sha256",
    "load_json_bytes",
    "snapshot_json",
]
