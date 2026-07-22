#!/usr/bin/env python3
"""Inspect and validate versioned RecClaw mechanism-space contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recclaw_core.mechanism_space import (  # noqa: E402
    CompileStatus,
    available_space_ids,
    catalog_digest,
    compile_program_bytes,
    program_schema,
    prompt_projection,
    space_identity,
    validate_development_binding_bytes,
)


def _emit(value: Any, *, pretty: bool) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
    )


def _report_exit(status: CompileStatus) -> int:
    if status in {CompileStatus.VALID_WIRED, CompileStatus.VALID_NEEDS_IMPLEMENTATION}:
        return 0
    if status == CompileStatus.INTERNAL_ERROR:
        return 3
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed CLI for RecClaw mechanism-space identity, prompt, and compile contracts"
    )
    parser.add_argument("--pretty", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    identity_parser = subparsers.add_parser("identity")
    identity_parser.add_argument("search_space_id")

    projection_parser = subparsers.add_parser("projection")
    projection_parser.add_argument("search_space_id")

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("search_space_id")
    schema_parser.add_argument("--profile-id")
    schema_parser.add_argument("--profile-digest")
    schema_parser.add_argument("--profile-kind", default="OFFLINE_TOPN")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("program", type=Path)

    binding_parser = subparsers.add_parser("validate-binding")
    binding_parser.add_argument("program", type=Path)
    binding_parser.add_argument("binding", type=Path)

    args = parser.parse_args()
    if args.command == "list":
        _emit(
            {
                "catalog_digest": catalog_digest(),
                "search_space_ids": list(available_space_ids()),
            },
            pretty=args.pretty,
        )
        return 0
    if args.command == "identity":
        _emit(space_identity(args.search_space_id).to_dict(), pretty=args.pretty)
        return 0
    if args.command == "projection":
        _emit(prompt_projection(args.search_space_id), pretty=args.pretty)
        return 0
    if args.command == "schema":
        schema = program_schema(args.search_space_id)
        provided_profile_fields = [args.profile_id, args.profile_digest]
        if any(provided_profile_fields) and not all(provided_profile_fields):
            parser.error("--profile-id and --profile-digest must be supplied together")
        if args.profile_id:
            if len(args.profile_digest) != 64 or any(
                char not in "0123456789abcdef" for char in args.profile_digest
            ):
                parser.error("--profile-digest must be 64 lowercase hexadecimal characters")
            schema["properties"]["profile_ref"] = {
                "const": {
                    "profile_id": args.profile_id,
                    "profile_digest": args.profile_digest,
                    "profile_kind": args.profile_kind,
                }
            }
        _emit(schema, pretty=args.pretty)
        return 0
    if args.command == "compile":
        report = compile_program_bytes(args.program.read_bytes())
        _emit(report.to_dict(), pretty=args.pretty)
        return _report_exit(report.status)
    if args.command == "validate-binding":
        report = validate_development_binding_bytes(
            args.program.read_bytes(),
            args.binding.read_bytes(),
        )
        _emit(report.to_dict(), pretty=args.pretty)
        return _report_exit(report.status)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
