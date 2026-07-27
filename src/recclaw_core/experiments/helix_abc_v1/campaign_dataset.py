"""Deterministic ML-1M development/held-out projection for Campaign runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .canonical import bytes_sha256, canonical_json_bytes, sha256_digest
from .runtime_contracts import DevelopmentRecSysProtocolV1


class CampaignDatasetError(ValueError):
    pass


_PROFILE_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "campaign_partition_profile_v1.json"
)
_PROTOCOL_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "campaign_development_protocol_v1.json"
)


def campaign_partition_profile() -> dict[str, Any]:
    value = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignDatasetError("partition profile must be an object")
    return value


def campaign_development_protocol() -> DevelopmentRecSysProtocolV1:
    return DevelopmentRecSysProtocolV1(
        json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    )


def _row_key(seed: int, user_id: str, row_index: int, row: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{user_id}\0{row_index}\0{row}".encode("utf-8")
    ).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partition_interactions(
    interaction_path: Path,
    *,
    seed: int,
) -> tuple[str, int, dict[str, list[tuple[int, str]]]]:
    raw_lines = interaction_path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise CampaignDatasetError("source interaction file is empty")
    header, rows = raw_lines[0], raw_lines[1:]
    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        user_id = row.split("\t", 1)[0]
        by_user[user_id].append((index, row))
    partitions: dict[str, list[tuple[int, str]]] = {
        "train": [],
        "dev": [],
        "heldout": [],
    }
    for user_rows in by_user.values():
        ordered = sorted(
            user_rows,
            key=lambda item: _row_key(
                seed, item[1].split("\t", 1)[0], *item
            ),
        )
        count = len(ordered)
        if count < 3:
            partitions["train"].extend(ordered)
            continue
        train_count = min(count - 2, max(1, int(count * 0.8)))
        remaining = count - train_count
        dev_count = max(1, remaining // 2)
        partitions["train"].extend(ordered[:train_count])
        partitions["dev"].extend(
            ordered[train_count : train_count + dev_count]
        )
        partitions["heldout"].extend(
            ordered[train_count + dev_count :]
        )
    for values in partitions.values():
        values.sort(key=lambda item: item[0])
    return header, len(rows), partitions


def _interaction_payload(
    header: str, values: list[tuple[int, str]]
) -> bytes:
    return (
        header + "\n" + "\n".join(row for _index, row in values) + "\n"
    ).encode("utf-8")


def inspect_materialized_campaign_dataset(
    *,
    source_dataset_root: Path,
    search_parent_root: Path,
    heldout_parent_root: Path,
) -> dict[str, Any]:
    """Re-derive the frozen split and compare every materialized byte."""

    profile = campaign_partition_profile()
    source = source_dataset_root.resolve()
    search_dataset = search_parent_root.resolve() / str(profile["dataset"])
    heldout_dataset = (
        heldout_parent_root.resolve() / str(profile["dataset"])
    )
    header, source_row_count, partitions = _partition_interactions(
        source / "ml-1m.inter",
        seed=int(profile["seed"]),
    )
    expected = {
        search_dataset / "ml-1m.train.inter": _interaction_payload(
            header, partitions["train"]
        ),
        search_dataset / "ml-1m.dev.inter": _interaction_payload(
            header, partitions["dev"]
        ),
        heldout_dataset / "ml-1m.heldout.inter": _interaction_payload(
            header, partitions["heldout"]
        ),
    }
    byte_exact = all(
        path.is_file() and path.read_bytes() == payload
        for path, payload in expected.items()
    )
    metadata_exact = all(
        (search_dataset / name).is_file()
        and (heldout_dataset / name).is_file()
        and (search_dataset / name).read_bytes()
        == (source / name).read_bytes()
        and (heldout_dataset / name).read_bytes()
        == (source / name).read_bytes()
        for name in ("ml-1m.item", "ml-1m.user")
    )
    row_id_sets = {
        name: {index for index, _row in values}
        for name, values in partitions.items()
    }
    zero_overlap = not bool(
        row_id_sets["train"] & row_id_sets["dev"]
        or row_id_sets["train"] & row_id_sets["heldout"]
        or row_id_sets["dev"] & row_id_sets["heldout"]
    )
    manifest = {
        "profile_digest": sha256_digest(profile),
        "source": {
            name: bytes_sha256((source / name).read_bytes())
            for name in ("ml-1m.inter", "ml-1m.item", "ml-1m.user")
        },
        "counts": {
            name: len(values) for name, values in partitions.items()
        },
        "search_root": search_dataset.as_posix(),
        "heldout_root": heldout_dataset.as_posix(),
        "search_files": {
            path.name: bytes_sha256(payload)
            for path, payload in expected.items()
            if path.parent == search_dataset
        },
        "heldout_files": {
            path.name: bytes_sha256(payload)
            for path, payload in expected.items()
            if path.parent == heldout_dataset
        },
        "zero_overlap": zero_overlap,
        "all_source_rows_assigned_once": (
            sum(len(values) for values in partitions.values())
            == source_row_count
        ),
    }
    manifest["manifest_digest"] = sha256_digest(manifest)
    return {
        **manifest,
        "interaction_bytes_exact": byte_exact,
        "metadata_bytes_exact": metadata_exact,
    }


def materialize_campaign_dataset(
    *,
    source_dataset_root: Path,
    search_parent_root: Path,
    heldout_parent_root: Path,
) -> dict[str, Any]:
    """Create fresh roots; the search root never contains held-out rows."""

    profile = campaign_partition_profile()
    source = source_dataset_root.resolve()
    search_dataset = search_parent_root.resolve() / str(profile["dataset"])
    heldout_dataset = heldout_parent_root.resolve() / str(profile["dataset"])
    if search_dataset.exists() or heldout_dataset.exists():
        raise CampaignDatasetError("campaign dataset roots must be fresh")
    header, source_row_count, partitions = _partition_interactions(
        source / "ml-1m.inter",
        seed=int(profile["seed"]),
    )
    search_dataset.mkdir(parents=True)
    heldout_dataset.mkdir(parents=True)
    search_files = {
        "ml-1m.train.inter": partitions["train"],
        "ml-1m.dev.inter": partitions["dev"],
    }
    heldout_files = {"ml-1m.heldout.inter": partitions["heldout"]}
    for name, values in {**search_files, **heldout_files}.items():
        target_root = (
            heldout_dataset if "heldout" in name else search_dataset
        )
        payload = _interaction_payload(header, values)
        _write_new(target_root / name, payload)
    for metadata_name in ("ml-1m.item", "ml-1m.user"):
        payload = (source / metadata_name).read_bytes()
        _write_new(search_dataset / metadata_name, payload)
        _write_new(heldout_dataset / metadata_name, payload)

    row_id_sets = {
        name: {index for index, _row in values}
        for name, values in partitions.items()
    }
    overlap = (
        row_id_sets["train"] & row_id_sets["dev"]
        or row_id_sets["train"] & row_id_sets["heldout"]
        or row_id_sets["dev"] & row_id_sets["heldout"]
    )
    manifest = {
        "profile_digest": sha256_digest(profile),
        "source": {
            name: bytes_sha256((source / name).read_bytes())
            for name in ("ml-1m.inter", "ml-1m.item", "ml-1m.user")
        },
        "counts": {
            name: len(values) for name, values in partitions.items()
        },
        "search_root": search_dataset.as_posix(),
        "heldout_root": heldout_dataset.as_posix(),
        "search_files": {
            name: bytes_sha256((search_dataset / name).read_bytes())
            for name in search_files
        },
        "heldout_files": {
            name: bytes_sha256((heldout_dataset / name).read_bytes())
            for name in heldout_files
        },
        "zero_overlap": not bool(overlap),
        "all_source_rows_assigned_once": (
            sum(len(values) for values in partitions.values())
            == source_row_count
        ),
    }
    manifest["manifest_digest"] = sha256_digest(manifest)
    _write_new(
        search_parent_root.resolve() / "search_partition_manifest.json",
        canonical_json_bytes(
            {
                "counts": {
                    "train": len(partitions["train"]),
                    "development_validation": len(partitions["dev"]),
                },
                "profile_digest": manifest["profile_digest"],
                "search_files": manifest["search_files"],
            }
        )
        + b"\n",
    )
    return manifest


__all__ = [
    "CampaignDatasetError",
    "campaign_partition_profile",
    "campaign_development_protocol",
    "materialize_campaign_dataset",
    "inspect_materialized_campaign_dataset",
]
