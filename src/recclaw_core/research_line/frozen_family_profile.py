"""Exact execution-profile binding for V2R4 declarative families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest
from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeMechanismSpaceProvider,
)


class FrozenFamilyProfileError(ValueError):
    pass


_SEARCH_DATA_FIELDS = {"root", "manifest_ref", "manifest_sha256"}
_SEARCH_MANIFEST_FIELDS = {
    "schema",
    "dataset",
    "parent_interaction_sha256",
    "parent_split",
    "partition_files",
    "abi_partition_roles",
    "heldout_partition_present",
}
_SEARCH_MANIFEST_OPTIONAL_FIELDS = {"item_feature"}
_PARTITION_FILE_FIELDS = {"path", "sha256"}
_ITEM_FEATURE_FIELDS = {"path", "sha256", "items"}


def _search_root_relative_path(*, root: Path, path: Path, role: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise FrozenFamilyProfileError(
            f"{role} asset escapes the search-only data root"
        ) from error


def _verify_sequential_benchmark_partitions(
    *,
    config_overrides: Mapping[str, Any],
    partition_assets: tuple[Mapping[str, Any], ...],
) -> None:
    if config_overrides.get("recclaw_model_type") != "SEQUENTIAL":
        return
    item_field = config_overrides.get("ITEM_ID_FIELD", "item_id")
    user_field = config_overrides.get("USER_ID_FIELD", "user_id")
    time_field = config_overrides.get("TIME_FIELD")
    list_suffix = config_overrides.get("LIST_SUFFIX", "_list")
    if not all(
        isinstance(value, str) and value
        for value in (item_field, user_field, time_field, list_suffix)
    ):
        raise FrozenFamilyProfileError(
            "sequential search data lacks the RecBole benchmark field ABI"
        )
    item_list_field = item_field + list_suffix
    time_list_field = time_field + list_suffix
    if config_overrides.get("alias_of_item_id") != [item_list_field]:
        raise FrozenFamilyProfileError(
            "sequential search data must alias item_id_list to item_id"
        )
    required_fields = {
        user_field: "token",
        item_field: "token",
        time_field: "float",
        item_list_field: "token_seq",
        time_list_field: "float_seq",
    }
    for asset in partition_assets:
        path = Path(str(asset["path"]))
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                header = stream.readline().rstrip("\r\n")
                first_row = stream.readline().rstrip("\r\n")
        except (OSError, UnicodeDecodeError) as error:
            raise FrozenFamilyProfileError(
                "sequential search partition is not valid UTF-8 benchmark data"
            ) from error
        columns = header.split("\t") if header else []
        field_types: dict[str, str] = {}
        for column in columns:
            field, separator, field_type = column.partition(":")
            if not separator or not field or not field_type or field in field_types:
                raise FrozenFamilyProfileError(
                    "sequential search partitions must use pre-augmented RecBole benchmark columns"
                )
            field_types[field] = field_type
        if any(
            field_types.get(field) != field_type
            for field, field_type in required_fields.items()
        ):
            raise FrozenFamilyProfileError(
                "sequential search partitions must use pre-augmented RecBole benchmark columns"
            )
        values = first_row.split("\t") if first_row else []
        if len(values) != len(columns):
            raise FrozenFamilyProfileError(
                "sequential search partitions must contain pre-augmented benchmark rows"
            )
        by_field = dict(zip((column.partition(":")[0] for column in columns), values))
        item_history = by_field[item_list_field].split()
        time_history = by_field[time_list_field].split()
        if not item_history or len(item_history) != len(time_history):
            raise FrozenFamilyProfileError(
                "sequential search partitions must contain aligned prefix histories"
            )


def _verify_exact_search_root(*, root: Path, assets: tuple[Mapping[str, Any], ...]) -> None:
    expected = {
        _search_root_relative_path(
            root=root,
            path=Path(str(asset["path"])),
            role=str(asset["role"]),
        )
        for asset in assets
    }
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FrozenFamilyProfileError(
                "search-only data root must not contain symbolic links"
            )
        if not path.is_file():
            continue
        _search_root_relative_path(root=root, path=path, role="SEARCH_ROOT")
        observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise FrozenFamilyProfileError(
            "search-only data root must physically contain only train and development "
            "interactions plus the declared manifest and auxiliary assets"
        )


def _verified_asset(
    *,
    role: str,
    path_value: Any,
    expected_sha256: Any,
) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise FrozenFamilyProfileError(f"{role} asset path is missing")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise FrozenFamilyProfileError(f"{role} frozen sha256 is invalid")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FrozenFamilyProfileError(f"{role} asset is missing: {path}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise FrozenFamilyProfileError(f"{role} asset sha256 differs from frozen profile")
    return {
        "role": role,
        "path": str(path),
        "sha256": actual,
        "size_bytes": len(payload),
    }


def verify_frozen_family_assets(
    *,
    frozen_fields: Mapping[str, Any],
    config_overrides: Mapping[str, Any],
    search_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and hash the exact files consumed by a frozen family run."""

    dataset = frozen_fields.get("dataset")
    snapshot = frozen_fields.get("dataset_snapshot")
    if not isinstance(dataset, str) or not dataset or not isinstance(snapshot, Mapping):
        raise FrozenFamilyProfileError("frozen dataset snapshot is incomplete")
    if set(search_data) != _SEARCH_DATA_FIELDS:
        raise FrozenFamilyProfileError("search_data fields differ from the launch ABI")
    root_value = search_data.get("root")
    manifest_ref = search_data.get("manifest_ref")
    manifest_sha256 = search_data.get("manifest_sha256")
    if not isinstance(root_value, str) or not root_value.strip():
        raise FrozenFamilyProfileError("search-only data root is missing")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FrozenFamilyProfileError("search-only data root is unavailable")
    manifest_asset = _verified_asset(
        role="SEARCH_DATA_MANIFEST",
        path_value=manifest_ref,
        expected_sha256=manifest_sha256,
    )
    _search_root_relative_path(
        root=root,
        path=Path(manifest_asset["path"]),
        role="SEARCH_DATA_MANIFEST",
    )
    try:
        manifest = json.loads(Path(manifest_asset["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenFamilyProfileError("search-only data manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) not in (
        _SEARCH_MANIFEST_FIELDS,
        _SEARCH_MANIFEST_FIELDS | _SEARCH_MANIFEST_OPTIONAL_FIELDS,
    ):
        raise FrozenFamilyProfileError("search-only data manifest fields are invalid")
    if (
        manifest.get("schema") != "recclaw.search-only-dataset.v1"
        or manifest.get("dataset") != dataset
        or manifest.get("parent_interaction_sha256")
        != snapshot.get("interaction_sha256")
        or canonical_value(manifest.get("parent_split"))
        != canonical_value(
            frozen_fields.get(
                "split", frozen_fields.get("chronological_split")
            )
        )
        or manifest.get("heldout_partition_present") is not False
        or manifest.get("abi_partition_roles")
        != {
            "train": "TRAIN",
            "valid": "DEVELOPMENT_VALIDATION",
            "test": "DEVELOPMENT_VALIDATION",
        }
    ):
        raise FrozenFamilyProfileError(
            "search-only data manifest differs from the frozen parent development protocol"
        )
    partition_files = manifest.get("partition_files")
    if not isinstance(partition_files, Mapping) or set(partition_files) != {
        "train",
        "development",
    }:
        raise FrozenFamilyProfileError("search-only partition files are incomplete")
    expected_relative_paths = {
        "train": f"{dataset}/{dataset}.train.inter",
        "development": f"{dataset}/{dataset}.dev.inter",
    }
    assets = [manifest_asset]
    partition_assets: list[dict[str, Any]] = []
    for role, relative_path in expected_relative_paths.items():
        row = partition_files.get(role)
        if (
            not isinstance(row, Mapping)
            or set(row) != _PARTITION_FILE_FIELDS
            or row.get("path") != relative_path
        ):
            raise FrozenFamilyProfileError(
                f"search-only {role} partition path differs from the RecBole ABI"
            )
        resolved = (root / relative_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise FrozenFamilyProfileError(
                f"search-only {role} partition escapes the data root"
            ) from error
        partition_asset = _verified_asset(
            role=f"SEARCH_{role.upper()}_INTERACTIONS",
            path_value=str(resolved),
            expected_sha256=row.get("sha256"),
        )
        partition_assets.append(partition_asset)
        assets.append(partition_asset)
    dataset_root = (root / dataset).resolve()
    observed_interactions = {
        path.resolve().relative_to(root).as_posix()
        for path in dataset_root.glob("*.inter")
        if path.is_file()
    }
    if observed_interactions != set(expected_relative_paths.values()):
        raise FrozenFamilyProfileError(
            "search-only data root must physically contain only train and development interactions"
        )
    item_feature = manifest.get("item_feature")
    if item_feature is not None:
        expected_item_path = f"{dataset}/{dataset}.item"
        if (
            not isinstance(item_feature, Mapping)
            or set(item_feature) != _ITEM_FEATURE_FIELDS
            or item_feature.get("path") != expected_item_path
            or item_feature.get("items") != snapshot.get("items")
        ):
            raise FrozenFamilyProfileError(
                "search-only item feature differs from the frozen catalog"
            )
        assets.append(
            _verified_asset(
                role="SEARCH_ITEM_FEATURE",
                path_value=str((root / expected_item_path).resolve()),
                expected_sha256=item_feature.get("sha256"),
            )
        )
    auxiliary_specs = (
        (
            "CONTENT_EMBEDDINGS",
            "content_asset_identity",
            "sha256",
            "recclaw_content_asset",
        ),
        (
            "STRUCTURED_FIELD_MANIFEST",
            "structured_field_asset_identity",
            "sha256",
            "recclaw_structured_field_manifest",
        ),
        (
            "CATALOG_ITEM_MAPPING",
            "item_id_mapping_identity",
            "catalog_items_sha256",
            "recclaw_catalog_mapping",
        ),
    )
    for role, frozen_key, digest_key, config_key in auxiliary_specs:
        identity = frozen_fields.get(frozen_key)
        if identity is None:
            continue
        if not isinstance(identity, Mapping):
            raise FrozenFamilyProfileError(f"{role} frozen identity is invalid")
        assets.append(
            _verified_asset(
                role=role,
                path_value=config_overrides.get(config_key),
                expected_sha256=identity.get(digest_key),
            )
        )
    _verify_sequential_benchmark_partitions(
        config_overrides=config_overrides,
        partition_assets=tuple(partition_assets),
    )
    rows = sorted(assets, key=lambda row: row["role"])
    _verify_exact_search_root(root=root, assets=tuple(rows))
    payload = canonical_value(
        {
            "schema": "recclaw.execution-assets.v1",
            "assets": rows,
            "search_data": {
                "data_path": str(root),
                "manifest_ref": manifest_asset["path"],
                "manifest_sha256": manifest_asset["sha256"],
                "benchmark_filename": ["train", "dev", "dev"],
                "partition_roles": dict(manifest["abi_partition_roles"]),
            },
        }
    )
    return canonical_value({**payload, "digest": sha256_digest(payload)})


def build_frozen_family_profile(
    provider: DeclarativeMechanismSpaceProvider,
    *,
    profile_id: str,
    frozen_fields: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(profile_id, str) or not profile_id:
        raise FrozenFamilyProfileError("profile_id must be non-empty")
    expected = set(provider.spec.frozen_protocol_fields)
    if set(frozen_fields) != expected:
        raise FrozenFamilyProfileError(
            "frozen profile fields differ; missing="
            + ",".join(sorted(expected - set(frozen_fields)))
            + "; extra="
            + ",".join(sorted(set(frozen_fields) - expected))
        )
    values = canonical_value(dict(frozen_fields))
    profile_kind = provider.spec.supported_profile_kinds[0]
    payload = canonical_value(
        {
            "schema": "recclaw.frozen-family-profile.v2r4",
            "profile_id": profile_id,
            "profile_kind": profile_kind,
            "search_space_id": provider.identity().search_space_id,
            "search_space_digest": provider.identity().search_space_digest,
            "frozen_fields": values,
        }
    )
    digest = sha256_digest(payload)
    profile_ref = {
        "profile_id": profile_id,
        "profile_digest": digest,
        "profile_kind": profile_kind,
    }
    config_binding = {
        "profile_kind": profile_kind,
        "dataset": values["dataset"],
        "protocol_digest": digest,
        "frozen_profile": payload,
    }
    return canonical_value(profile_ref), canonical_value(config_binding)


def validate_frozen_family_profile(
    provider: DeclarativeMechanismSpaceProvider,
    *,
    profile_ref: Mapping[str, Any],
    execution_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(profile_ref, Mapping) or not isinstance(execution_config, Mapping):
        raise FrozenFamilyProfileError("profile_ref and execution_config must be mappings")
    frozen = execution_config.get("frozen_profile")
    if not isinstance(frozen, Mapping):
        raise FrozenFamilyProfileError("execution config lacks frozen_profile")
    expected_keys = {
        "schema",
        "profile_id",
        "profile_kind",
        "search_space_id",
        "search_space_digest",
        "frozen_fields",
    }
    if set(frozen) != expected_keys or frozen.get("schema") != "recclaw.frozen-family-profile.v2r4":
        raise FrozenFamilyProfileError("frozen_profile envelope is malformed")
    identity = provider.identity()
    if (
        frozen.get("search_space_id") != identity.search_space_id
        or frozen.get("search_space_digest") != identity.search_space_digest
        or frozen.get("profile_kind") not in provider.spec.supported_profile_kinds
    ):
        raise FrozenFamilyProfileError("frozen profile targets a different family identity")
    fields = frozen.get("frozen_fields")
    if not isinstance(fields, Mapping) or set(fields) != set(provider.spec.frozen_protocol_fields):
        raise FrozenFamilyProfileError("frozen profile does not bind every protocol field exactly")
    digest = sha256_digest(canonical_value(dict(frozen)))
    expected_ref = {
        "profile_id": frozen["profile_id"],
        "profile_digest": digest,
        "profile_kind": frozen["profile_kind"],
    }
    if canonical_value(dict(profile_ref)) != canonical_value(expected_ref):
        raise FrozenFamilyProfileError("mechanism program profile_ref does not bind frozen_profile")
    if execution_config.get("protocol_digest") != digest:
        raise FrozenFamilyProfileError("execution protocol_digest differs from frozen_profile")
    if execution_config.get("profile_kind") != frozen["profile_kind"]:
        raise FrozenFamilyProfileError("execution profile_kind differs from frozen_profile")
    if execution_config.get("dataset") != fields.get("dataset"):
        raise FrozenFamilyProfileError("execution dataset differs from frozen_profile")
    return canonical_value(dict(frozen))


__all__ = [
    "FrozenFamilyProfileError",
    "build_frozen_family_profile",
    "validate_frozen_family_profile",
    "verify_frozen_family_assets",
]
