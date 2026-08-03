"""Stdlib-only runtime binding and execution ownership bootstrap.

This module must remain importable before :mod:`recclaw_core` so that the
import-time PROJECTS_ROOT/SEARCH_DATA_ROOT/RECBole_ROOT constants are bound
from the manifest before any path-sensitive research module is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEPLOYMENT_SCHEMA = "recclaw.research-line.q5a-deployment-manifest.v1"
PREFLIGHT_SCHEMA = "recclaw.research-line.q5a-comprehensive-preflight.v1"
RUNTIME_BINDING_SCHEMA = "recclaw.research-line.q5a-runtime-binding.v1"
EXECUTION_GATE_SCHEMA = "recclaw.research-line.q5a-execution-gate.v1"
EXECUTION_OWNER_SCHEMA = "recclaw.research-line.q5a-execution-owner.v1"


class RuntimeBindingError(RuntimeError):
    """A manifest, runtime binding, preflight, or owner contract drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bytes_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeBindingError(f"JSON root is not an object: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], field: str, label: str) -> str:
    observed = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if not isinstance(observed, str) or observed != _digest(payload):
        raise RuntimeBindingError(f"{label} digest mismatch")
    return observed


def _resolve_record(record: Mapping[str, Any], repo_root: Path) -> tuple[Path, bool]:
    path = Path(str(record.get("path", "")))
    if path.is_file():
        return path.resolve(), False
    relative = record.get("relative_path")
    if isinstance(relative, str) and relative:
        relocated = (repo_root / relative).resolve()
        if relocated.is_file():
            return relocated, True
    raise RuntimeBindingError(f"runtime dependency is missing: {path}")


def _all_records(value: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if {"role", "sha256", "path"}.issubset(value):
            records.append(value)
        for child in value.values():
            records.extend(_all_records(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            records.extend(_all_records(child))
    return records


def _git_head(path: Path) -> str | None:
    head = path / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = path / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else value
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    supplied = payload.pop(digest_field, None)
    computed = _digest(payload)
    if supplied is not None and supplied != computed:
        raise RuntimeBindingError(f"{digest_field} mismatch while writing {path}")
    payload[digest_field] = computed
    with path.open("xb") as handle:
        handle.write(_canonical(payload))
    return payload


@dataclass(frozen=True)
class RuntimeBindingV1:
    deployment_digest: str
    binding_digest: str
    repo_root: Path
    projects_root: Path
    search_data_root: Path
    recbole_root: Path
    python_executable: Path
    api_config: Path
    preflight_root: Path
    campaign_root: Path
    prefreeze_digest: str
    source_tree_digest: str
    foundation_commit: str
    foundation_package_digest: str
    relocated_records: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest_path: Path, *, repo_root: Path | None = None) -> "RuntimeBindingV1":
        manifest_path = manifest_path.resolve()
        manifest = _load(manifest_path)
        if manifest.get("schema") != DEPLOYMENT_SCHEMA:
            raise RuntimeBindingError("deployment manifest schema drift")
        deployment_digest = _verify_self_digest(manifest, "deployment_digest", "deployment manifest")
        root = (repo_root or manifest_path.parent).resolve()
        records = _all_records(manifest)
        relocated: list[str] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            key = (str(record.get("role")), str(record.get("path")))
            if key in seen:
                continue
            seen.add(key)
            path, used_relocation = _resolve_record(record, root)
            if _bytes_digest(path) != str(record.get("sha256")):
                raise RuntimeBindingError(f"runtime dependency hash drift: {record.get('role')}")
            if used_relocation:
                relocated.append(str(record.get("role")))

        frozen = manifest.get("frozen_inputs")
        if not isinstance(frozen, Mapping):
            raise RuntimeBindingError("deployment manifest lacks frozen_inputs")
        for field in ("prefreeze_digest", "source_tree_digest", "foundation_package_digest"):
            value = frozen.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeBindingError(f"deployment frozen input is missing {field}")
        foundation_commit = frozen.get("foundation_commit")
        if not isinstance(foundation_commit, str) or len(foundation_commit) != 40:
            raise RuntimeBindingError("deployment foundation commit is invalid")

        runtime = manifest.get("runtime")
        provider = manifest.get("provider")
        accepted = manifest.get("accepted_external_roots")
        execution = manifest.get("execution_binding")
        if not all(isinstance(value, Mapping) for value in (runtime, provider, accepted, execution)):
            raise RuntimeBindingError("deployment runtime/execution bindings are incomplete")
        python_record = runtime["python_executable"]
        config_record = provider["config_reference"]
        search_record = next((record for record in records if record.get("role") == "search_partition_manifest"), None)
        if not isinstance(python_record, Mapping) or not isinstance(config_record, Mapping) or not isinstance(search_record, Mapping):
            raise RuntimeBindingError("deployment runtime file records are incomplete")
        python_path, _ = _resolve_record(python_record, root)
        api_path, _ = _resolve_record(config_record, root)
        search_path, _ = _resolve_record(search_record, root)
        projects_root = Path(str(accepted["projects_root"])).resolve()
        search_root = Path(str(runtime["search_data_root"])).resolve()
        recbole_root = Path(str(runtime["recbole_root"])).resolve()
        preflight_root = Path(str(execution["preflight_root"])).resolve()
        campaign_root = Path(str(execution["campaign_root"])).resolve()
        if not projects_root.is_dir() or not search_root.is_dir() or not recbole_root.is_dir():
            raise RuntimeBindingError("runtime root binding is not a directory")
        if search_path.resolve() != (search_root / "search_partition_manifest.json").resolve():
            raise RuntimeBindingError("search manifest path is not bound to search_data_root")
        if str(runtime.get("search_manifest_digest")) != str(search_record["sha256"]):
            raise RuntimeBindingError("search manifest recorded digest drift")
        if api_path.resolve() != Path(str(provider["config_reference"]["path"])).resolve() and not provider["config_reference"].get("relative_path"):
            raise RuntimeBindingError("Provider config path binding drift")
        expected_search = str(runtime.get("expected_search_manifest_digest"))
        if expected_search != str(search_record["sha256"]):
            raise RuntimeBindingError("search manifest digest binding drift")
        if str(provider.get("endpoint_digest")) != str(config_record.get("sha256")):
            raise RuntimeBindingError("Provider endpoint digest binding drift")
        expected_recbole = str(runtime.get("expected_recbole_commit"))
        if str(runtime.get("recbole_head")) != expected_recbole or _git_head(recbole_root) != expected_recbole:
            raise RuntimeBindingError("RecBole commit binding drift")
        expected_external = projects_root / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
        if Path(str(accepted.get("r1_external_root"))).resolve() != expected_external:
            raise RuntimeBindingError("accepted R1 external root binding drift")
        external_record = next((record for record in records if record.get("role") == "accepted_r1_external_receipt"), None)
        if not isinstance(external_record, Mapping):
            raise RuntimeBindingError("accepted R1 external receipt record is missing")
        external_path, _ = _resolve_record(external_record, root)
        if external_path != (expected_external / "R1_CANONICAL_RECEIPT.json").resolve():
            raise RuntimeBindingError("accepted R1 external receipt path binding drift")
        prefreeze_record = execution.get("prefreeze_manifest")
        if not isinstance(prefreeze_record, Mapping) or prefreeze_record.get("role") != "prefreeze_manifest":
            raise RuntimeBindingError("prefreeze execution binding record is missing")
        binding_payload = {
            "schema": RUNTIME_BINDING_SCHEMA,
            "deployment_digest": deployment_digest,
            "projects_root": str(projects_root),
            "search_data_root": str(search_root),
            "recbole_root": str(recbole_root),
            "python_executable": str(python_path),
            "api_config": str(api_path),
            "prefreeze_digest": frozen["prefreeze_digest"],
            "source_tree_digest": frozen["source_tree_digest"],
            "foundation_commit": foundation_commit,
            "foundation_package_digest": frozen["foundation_package_digest"],
            "preflight_root": str(preflight_root),
            "campaign_root": str(campaign_root),
        }
        return cls(
            deployment_digest=deployment_digest,
            binding_digest=_digest(binding_payload),
            repo_root=root,
            projects_root=projects_root,
            search_data_root=search_root,
            recbole_root=recbole_root,
            python_executable=python_path,
            api_config=api_path,
            preflight_root=preflight_root,
            campaign_root=campaign_root,
            prefreeze_digest=str(frozen["prefreeze_digest"]),
            source_tree_digest=str(frozen["source_tree_digest"]),
            foundation_commit=str(foundation_commit),
            foundation_package_digest=str(frozen["foundation_package_digest"]),
            relocated_records=tuple(sorted(set(relocated))),
        )

    def activate(self) -> "RuntimeBindingV1":
        """Activate environment and import paths before path-sensitive imports."""

        os.environ["RECCLAW_PROJECTS_ROOT"] = str(self.projects_root)
        os.environ["RECCLAW_SEARCH_DATA_ROOT"] = str(self.search_data_root)
        os.environ["RECCLAW_RECBOLE_ROOT"] = str(self.recbole_root)
        os.environ["RECCLAW_PYTHON_EXECUTABLE"] = str(self.python_executable)
        os.environ["RECCLAW_API_CONFIG"] = str(self.api_config)
        os.environ["RECCLAW_RUNTIME_BINDING_DIGEST"] = self.binding_digest
        for path in (self.repo_root, self.repo_root / "src", self.repo_root / "scripts", self.recbole_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        return self


def read_verified_preflight(receipt_path: Path, binding: RuntimeBindingV1) -> dict[str, Any]:
    receipt = _load(receipt_path.resolve())
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("status") != "PASS":
        raise RuntimeBindingError("preflight receipt is absent or not PASS")
    receipt_digest = _verify_self_digest(receipt, "preflight_digest", "preflight receipt")
    if receipt.get("deployment_digest") != binding.deployment_digest or receipt.get("binding_digest") != binding.binding_digest:
        raise RuntimeBindingError("preflight receipt deployment/binding digest drift")
    if receipt.get("source_tree_digest") != binding.source_tree_digest or receipt.get("prefreeze_digest") != binding.prefreeze_digest:
        raise RuntimeBindingError("preflight receipt frozen identity drift")
    if receipt.get("preflight_root") != str(binding.preflight_root):
        raise RuntimeBindingError("preflight receipt root is stale or relocated")
    if receipt.get("campaign_root") != str(binding.campaign_root):
        raise RuntimeBindingError("preflight receipt campaign root is stale or relocated")
    if any(int(receipt.get(field, -1)) != 0 for field in ("provider_calls", "implementer_calls", "qualification_calls", "training_runs", "held_out_reads", "retries")):
        raise RuntimeBindingError("preflight receipt contains a physical call, retry, or held-out read")
    if not receipt_digest:
        raise RuntimeBindingError("preflight receipt digest is empty")
    return receipt


def _physical_attempts(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.rglob("physical_attempt_01")) if path.is_dir()]


def claim_execution_owner(binding: RuntimeBindingV1, *, gate_receipt_path: Path) -> dict[str, Any]:
    root = binding.campaign_root
    if _physical_attempts(root):
        raise RuntimeBindingError("campaign root already contains physical attempt directories")
    allowed_prefreeze = {"PREFREEZE_MANIFEST.json", "POOL_GENERATION_PLAN.json"}
    if root.exists() and any(path.name not in allowed_prefreeze for path in root.iterdir()):
        raise RuntimeBindingError("campaign root contains non-prefreeze campaign state")
    owner_path = root / "EXECUTION_OWNER.json"
    if owner_path.exists():
        raise RuntimeBindingError("campaign root already has an execution owner")
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": EXECUTION_OWNER_SCHEMA,
        "owner_pid": os.getpid(),
        "owner_token": uuid.uuid4().hex,
        "campaign_root": str(root),
        "deployment_digest": binding.deployment_digest,
        "binding_digest": binding.binding_digest,
        "gate_receipt": str(gate_receipt_path.resolve()),
        "status": "CLAIMED",
    }
    return _write_exclusive(owner_path, payload, "owner_digest")


def verify_execution_owner(binding: RuntimeBindingV1, *, owner_token: str | None = None) -> dict[str, Any]:
    owner_path = binding.campaign_root / "EXECUTION_OWNER.json"
    if not owner_path.is_file():
        raise RuntimeBindingError("execution owner is missing")
    owner = _load(owner_path)
    _verify_self_digest(owner, "owner_digest", "execution owner")
    if owner.get("owner_pid") != os.getpid() or owner.get("campaign_root") != str(binding.campaign_root):
        raise RuntimeBindingError("execution owner PID/root drift")
    if owner_token is not None and owner.get("owner_token") != owner_token:
        raise RuntimeBindingError("execution owner token drift")
    if owner.get("deployment_digest") != binding.deployment_digest or owner.get("binding_digest") != binding.binding_digest:
        raise RuntimeBindingError("execution owner binding drift")
    return owner


def release_execution_owner(binding: RuntimeBindingV1, owner: Mapping[str, Any], *, status: str) -> None:
    owner_path = binding.campaign_root / "EXECUTION_OWNER.json"
    if not owner_path.is_file():
        raise RuntimeBindingError("execution owner disappeared")
    observed = _load(owner_path)
    if observed.get("owner_pid") != owner.get("owner_pid") or observed.get("owner_token") != owner.get("owner_token"):
        raise RuntimeBindingError("execution owner identity drift")
    released = {
        **owner,
        "status": status,
        "released_pid": os.getpid(),
    }
    release_path = binding.campaign_root / "EXECUTION_OWNER_RELEASED.json"
    with release_path.open("xb") as handle:
        handle.write(_canonical({**released, "release_digest": _digest(released)}))
    owner_path.unlink()


def verify_execution_gate(
    *,
    binding: RuntimeBindingV1,
    preflight_receipt_path: Path,
    prefreeze_manifest_path: Path,
) -> dict[str, Any]:
    preflight = read_verified_preflight(preflight_receipt_path, binding)
    prefreeze = _load(prefreeze_manifest_path.resolve())
    if prefreeze.get("prefreeze_digest") != binding.prefreeze_digest:
        raise RuntimeBindingError("prefreeze digest does not match deployment binding")
    prefreeze_payload = dict(prefreeze)
    observed_prefreeze_digest = prefreeze_payload.pop("prefreeze_digest", None)
    if observed_prefreeze_digest != _digest(prefreeze_payload):
        raise RuntimeBindingError("prefreeze manifest self-digest mismatch")
    if prefreeze.get("source_tree_digest") != binding.source_tree_digest:
        raise RuntimeBindingError("source tree digest does not match deployment binding")
    payload = {
        "schema": EXECUTION_GATE_SCHEMA,
        "status": "PASS",
        "deployment_digest": binding.deployment_digest,
        "binding_digest": binding.binding_digest,
        "prefreeze_digest": binding.prefreeze_digest,
        "source_tree_digest": binding.source_tree_digest,
        "foundation_commit": binding.foundation_commit,
        "foundation_package_digest": binding.foundation_package_digest,
        "preflight_digest": preflight["preflight_digest"],
        "preflight_root": str(binding.preflight_root),
        "campaign_root": str(binding.campaign_root),
        "owner_pid": os.getpid(),
        "provider_calls_before_gate": 0,
        "held_out_reads": 0,
        "retries": 0,
    }
    return {**payload, "gate_digest": _digest(payload)}


def write_execution_gate(path: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    if gate.get("schema") != EXECUTION_GATE_SCHEMA or gate.get("status") != "PASS":
        raise RuntimeBindingError("only a PASS execution gate may be written")
    return _write_exclusive(path, gate, "gate_digest")


__all__ = [
    "EXECUTION_GATE_SCHEMA",
    "EXECUTION_OWNER_SCHEMA",
    "RuntimeBindingError",
    "RuntimeBindingV1",
    "claim_execution_owner",
    "read_verified_preflight",
    "release_execution_owner",
    "verify_execution_owner",
    "verify_execution_gate",
    "write_execution_gate",
]
