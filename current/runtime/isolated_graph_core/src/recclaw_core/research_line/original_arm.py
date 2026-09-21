"""Stateful production arm around the exact pinned pre-Research-Line RecClaw."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, ClassVar, Protocol

import yaml


PINNED_ORIGINAL_COMMIT = "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
PINNED_ORIGINAL_FILE_COUNT = 78
PINNED_ORIGINAL_TREE = "3f9049509e5e09ae59a0d6aba79a5c2094dd3c2c"
PINNED_ORIGINAL_ARCHIVE_SHA256 = "92db7f87215ae7a21d422847825a6970cb86df5f361c41aba10513cab4cc08ce"
FINAL_SELF_REFLECTION_DIRECTIVE = (
    "RecClaw algorithm-first run: keep the frozen ML-1M full-sort general-rec "
    "protocol unchanged; prioritize LLM-driven algorithm discovery, mechanism "
    "composition, local extension implementation, smoke verification, and formal "
    "runs. Parameter-only tuning is allowed only as a small sanity/refinement "
    "budget after credible algorithm signal. Do not propose sequential "
    "recommendation in this experiment line."
)


class OriginalArmError(RuntimeError):
    pass


class OriginalArmSourceError(OriginalArmError):
    pass


class OriginalArmStateError(OriginalArmError):
    pass


class OriginalOutcomeClass(str, Enum):
    SUCCESS = "SUCCESS"
    CRASH = "CRASH"
    OUTCOME_MISSING = "OUTCOME_MISSING"
    CENSORED = "CENSORED"


class LLMResponder(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        schema_name: str,
        response_schema: Mapping[str, Any],
    ) -> str | Mapping[str, Any] | Sequence[Any]: ...


class ExperimentRunner(Protocol):
    """Common family-neutral boundary for the physical experiment."""

    def __call__(
        self,
        candidate: Mapping[str, Any],
        params: Mapping[str, Any],
        round_index: int,
        run_root: Path,
    ) -> Mapping[str, Any]: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class PinnedOriginalSourceV1:
    """Materialize the complete pinned tree without importing checkout helpers."""

    # The pinned runner's registry is the one arm-local mutable source file.
    # Candidate implementation and source files remain byte-bound to the
    # pinned archive; only this registry is extended by the Original loop.
    mutable_archive_members: ClassVar[frozenset[str]] = frozenset(
        {"configs/candidate_registry.yaml"}
    )

    repository_root: Path
    materialization_root: Path
    tree_id: str = field(default="", init=False)
    archive_sha256: str = field(default="", init=False)
    _agent_module: ModuleType | None = field(default=None, init=False)
    _archive_bytes: bytes = field(default=b"", init=False, repr=False)

    def __post_init__(self) -> None:
        self.repository_root = Path(self.repository_root).resolve()
        self.materialization_root = Path(self.materialization_root).resolve()

    def _git(self, *args: str) -> bytes:
        safe = str(self.repository_root).replace("\\", "/")
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={safe}", "-C", str(self.repository_root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            raise OriginalArmSourceError(
                f"pinned Git read failed ({completed.returncode}): "
                f"{' '.join(args)}: {completed.stderr.decode(errors='replace').strip()}"
            )
        return completed.stdout

    @staticmethod
    def _target(root: Path, name: str) -> Path:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise OriginalArmSourceError(f"unsafe pinned archive path: {name}")
        target = root.joinpath(*relative.parts)
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise OriginalArmSourceError(f"archive escapes arm root: {name}") from error
        return target

    def materialize(self) -> None:
        if self.tree_id:
            return
        archive_path = os.environ.get("RECCLAW_PINNED_ORIGINAL_ARCHIVE")
        if archive_path:
            archive = Path(archive_path).resolve().read_bytes()
            self.tree_id = PINNED_ORIGINAL_TREE
        else:
            self.tree_id = self._git(
                "rev-parse", f"{PINNED_ORIGINAL_COMMIT}^{{tree}}"
            ).decode("ascii").strip()
            archive = self._git("archive", "--format=tar", PINNED_ORIGINAL_COMMIT)
        self.archive_sha256 = hashlib.sha256(archive).hexdigest()
        if self.tree_id != PINNED_ORIGINAL_TREE or self.archive_sha256 != PINNED_ORIGINAL_ARCHIVE_SHA256:
            raise OriginalArmSourceError("pinned Original archive identity mismatch")
        self._archive_bytes = archive
        self.materialization_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            files = [member for member in bundle.getmembers() if member.isfile()]
            if len(files) != PINNED_ORIGINAL_FILE_COUNT:
                raise OriginalArmSourceError(
                    "pinned archive file count mismatch: "
                    f"expected {PINNED_ORIGINAL_FILE_COUNT}, got {len(files)}"
                )
            for member in files:
                target = self._target(self.materialization_root, member.name)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise OriginalArmSourceError(
                        f"pinned archive member is unreadable: {member.name}"
                    )
                payload = extracted.read()
                if target.exists() or target.is_symlink():
                    if not target.is_file():
                        raise OriginalArmSourceError(
                            f"materialized pinned file differs: {member.name}"
                        )
                    if member.name in self.mutable_archive_members:
                        continue
                    if target.read_bytes() != payload:
                        raise OriginalArmSourceError(
                            f"materialized pinned file differs: {member.name}"
                        )
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)

    def archive_member_bytes(self, name: str) -> bytes:
        self.materialize()
        with tarfile.open(fileobj=io.BytesIO(self._archive_bytes), mode="r:") as bundle:
            member = bundle.getmember(name)
            extracted = bundle.extractfile(member)
            if extracted is None or not member.isfile():
                raise OriginalArmSourceError(f"pinned archive member is unreadable: {name}")
            return extracted.read()

    @property
    def identity_digest(self) -> str:
        self.materialize()
        return _digest(
            {
                "release": "PinnedOriginalSourceV1",
                "commit": PINNED_ORIGINAL_COMMIT,
                "tree": self.tree_id,
                "archive_sha256": self.archive_sha256,
                "file_count": PINNED_ORIGINAL_FILE_COUNT,
            }
        )

    @staticmethod
    def _load(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise OriginalArmSourceError(f"cannot load pinned module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module

    def load_agent(self) -> ModuleType:
        if self._agent_module is not None:
            return self._agent_module
        self.materialize()
        scripts = self.materialization_root / "scripts"
        names = ("action_space", "collect_result", "compare_runs")
        prior = {name: sys.modules.get(name) for name in names}
        loaded: list[str] = []
        try:
            for name in names:
                sys.modules.pop(name, None)
                self._load(name, scripts / f"{name}.py")
                loaded.append(name)
            self._agent_module = self._load(
                "_recclaw_pinned_original_" + self.identity_digest[:16],
                scripts / "agent.py",
            )
        except Exception as error:
            raise OriginalArmSourceError(
                f"could not load pinned RecClawAgent: {type(error).__name__}: {error}"
            ) from error
        finally:
            for name in loaded:
                if prior[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior[name]
        return self._agent_module


@dataclass(frozen=True, slots=True)
class OriginalArmRoundResultV1:
    schema: ClassVar[str] = "recclaw.original-arm.round-result.v1"
    arm_id: str
    round_index: int
    resumed: bool
    execution_opportunities: int
    execution_calls: int
    selected_candidate: Mapping[str, Any]
    selected_params: Mapping[str, Any]
    candidate_id: str
    run_id: str
    outcome_class: str
    candidate_result: Mapping[str, Any]
    compare_baseline: Mapping[str, Any]
    compare_history_best: Mapping[str, Any]
    dimension_report: Mapping[str, Any]
    decision: str
    reason: str
    next_action: str
    proposal_count: int
    memory_digest: str
    source_release_digest: str
    state_projection: Mapping[str, Any]
    state_paths: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, **_jsonable(asdict(self))}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OriginalArmRoundResultV1":
        if payload.get("schema") not in {None, cls.schema}:
            raise OriginalArmStateError(f"unsupported round result schema: {payload.get('schema')}")
        values = dict(payload)
        values.pop("schema", None)
        for name in (
            "selected_candidate",
            "selected_params",
            "candidate_result",
            "compare_baseline",
            "compare_history_best",
            "dimension_report",
            "state_projection",
            "state_paths",
        ):
            values[name] = dict(values.get(name) or {})
        return cls(**values)

    def resumed_result(self) -> "OriginalArmRoundResultV1":
        return replace(self, resumed=True, execution_calls=0)


@dataclass(slots=True)
class PinnedOriginalArmV1:
    """One arm-local pinned Original campaign with one opportunity per round."""

    repository_root: Path
    campaign_root: Path
    initial_executable_space: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    protocol: Mapping[str, Any] = field(default_factory=dict)
    search_seed: int = 42
    proposal_every: int = 3
    proposal_count: int = 6
    llm_responder: LLMResponder | None = None
    experiment_runner: ExperimentRunner | None = None
    checkpoint_policy: str = "cleanup_all"
    loop_mode: str = "auto"
    search_intensity: str = "algorithm_first"
    max_pending_implemented: int = 6
    max_implement_per_round: int = 2
    algorithm_first_explore_rounds: int = 20
    seed_validation_min_metric: float = 0.274
    refresh_experience_every: int = 10
    static_experiment_directive: str = FINAL_SELF_REFLECTION_DIRECTIVE
    auto_implement_code_required: bool = True
    memory_read_limit: int = 2000
    prompt_memory_tail: int = 120
    llm_runtime: Mapping[str, str] = field(default_factory=dict)
    use_native_registry: bool = False

    source: PinnedOriginalSourceV1 = field(init=False)
    _agent_module: ModuleType = field(init=False)
    _agent: Any = field(init=False)
    _candidates: tuple[dict[str, Any], ...] = field(init=False)
    _state: Path = field(init=False)
    _results: Path = field(init=False)
    _registry: Path = field(init=False)
    _memory: Path = field(init=False)
    _results_csv: Path = field(init=False)
    _baseline: Path = field(init=False)
    _proposals: Path = field(init=False)
    _rounds: Path = field(init=False)
    llm_call_count: int = field(default=0, init=False)
    _last_planner_negotiation: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.repository_root = Path(self.repository_root).resolve()
        self.campaign_root = Path(self.campaign_root).resolve()
        if self.repository_root == self.campaign_root:
            raise OriginalArmStateError("campaign_root must not be the source repository")
        self.protocol = dict(self.protocol)
        self.llm_runtime = dict(self.llm_runtime)
        self._candidates = self._normalize_candidates(self.initial_executable_space)
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        self._bind_pinned_recbole_root()
        self._state = self.campaign_root / "original_state"
        self._results = self._state / "results"
        self._memory = self._state / "history" / "agent_memory.jsonl"
        self._results_csv = self._results / "results.csv"
        self._baseline = self._results / "baseline"
        self._proposals = self._results / "candidate_proposals.jsonl"
        self._rounds = self._state / "rounds"
        self.source = PinnedOriginalSourceV1(
            self.repository_root,
            self.campaign_root / "pinned_original_source",
        )
        self.source.materialize()
        # The pinned Original implementer shells out to run_candidate.py, whose
        # production contract reads PROJECT_ROOT/configs/candidate_registry.yaml.
        # Its materialized source is already arm-local, so keep the active
        # registry there as Original expects instead of splitting one arm over
        # two registries.
        self._registry = self.source_root / "configs" / "candidate_registry.yaml"
        if self.use_native_registry:
            try:
                native_registry = yaml.safe_load(
                    self.source.archive_member_bytes(
                        "configs/candidate_registry.yaml"
                    ).decode("utf-8")
                )
            except (UnicodeDecodeError, yaml.YAMLError, KeyError) as error:
                raise OriginalArmSourceError(
                    "pinned Original native registry cannot be loaded"
                ) from error
            self._candidates = self._normalize_candidates(
                native_registry,
                allowed_runner_types={"config_only", "model", "posthoc"},
            )
            self.initial_executable_space = {
                "profile_ref": "RECCLAW_MAIN_NATIVE_REGISTRY_V1",
                "profile_digest": _digest(native_registry),
                "candidates": list(self._candidates),
            }
        self._initialize_state()
        self._agent_module = self.source.load_agent()
        self._agent = self._make_agent()
        self._install_llm()

    def _bind_pinned_recbole_root(self) -> None:
        """Expose the canonical runtime binding to the pinned subprocesses."""

        # fresh_r1 and the paired runtime carry the frozen dependency under
        # RECCLAW_RECBOLE_ROOT.  The pinned archive predates that boundary and
        # run_candidate.py only consumes the legacy RECBOLE_ROOT name.
        recbole_root = os.environ.get("RECCLAW_RECBOLE_ROOT")
        if not recbole_root:
            return
        canonical_root = Path(recbole_root).resolve()
        data_root = os.environ.get("RECCLAW_SEARCH_DATA_ROOT")
        if not data_root:
            os.environ["RECBOLE_ROOT"] = str(canonical_root)
            return

        # The pinned run_candidate.py also fixes data_path to
        # RECBOLE_ROOT/dataset.  Keep that legacy path arm-local and project
        # only the train-backed random-split input plus its non-outcome
        # metadata.  Do not expose the frozen evaluator's dev/heldout files.
        view_root = self.campaign_root / "original_runtime" / "recbole_root"
        view_root.mkdir(parents=True, exist_ok=True)
        links = {
            "recbole": canonical_root / "recbole",
            "run_recbole.py": canonical_root / "run_recbole.py",
        }
        for name, target in links.items():
            link = view_root / name
            if link.is_symlink() or link.exists():
                if link.resolve() != target:
                    raise OriginalArmStateError(
                        f"pinned RecBole compatibility path differs: {name}"
                    )
                continue
            link.symlink_to(target, target_is_directory=target.is_dir())

        frozen_dataset_root = Path(data_root).resolve()
        frozen_ml1m_root = frozen_dataset_root / "ml-1m"
        if not frozen_ml1m_root.is_dir():
            raise OriginalArmStateError(
                f"frozen evaluator dataset is missing ml-1m: {frozen_ml1m_root}"
            )

        compatibility_dataset_root = view_root / "dataset"
        if compatibility_dataset_root.is_symlink():
            if compatibility_dataset_root.resolve() != frozen_dataset_root:
                raise OriginalArmStateError(
                    "pinned RecBole compatibility dataset path differs"
                )
            # Migrate the prior whole-root compatibility link without touching
            # the frozen evaluator root it referenced.
            compatibility_dataset_root.unlink()
        elif compatibility_dataset_root.exists() and not compatibility_dataset_root.is_dir():
            raise OriginalArmStateError(
                f"pinned RecBole compatibility dataset is not a directory: "
                f"{compatibility_dataset_root}"
            )
        compatibility_ml1m_root = compatibility_dataset_root / "ml-1m"
        compatibility_ml1m_root.mkdir(parents=True, exist_ok=True)
        projected_files = {
            "ml-1m.inter": frozen_ml1m_root / "ml-1m.train.inter",
            "ml-1m.user": frozen_ml1m_root / "ml-1m.user",
            "ml-1m.item": frozen_ml1m_root / "ml-1m.item",
        }
        for name, target in projected_files.items():
            target = target.resolve()
            if not target.is_file():
                raise OriginalArmStateError(
                    f"frozen evaluator file is missing: {target}"
                )
            link = compatibility_ml1m_root / name
            if link.is_symlink() or link.exists():
                if link.resolve() != target:
                    raise OriginalArmStateError(
                        f"pinned RecBole compatibility data path differs: {name}"
                    )
                continue
            link.symlink_to(target)
        exposed = {path.name for path in compatibility_ml1m_root.iterdir()}
        if exposed != set(projected_files):
            raise OriginalArmStateError(
                "pinned RecBole compatibility dataset exposes unexpected files: "
                + ", ".join(sorted(exposed - set(projected_files)))
            )
        os.environ["RECBOLE_ROOT"] = str(view_root)

    @staticmethod
    def _normalize_candidates(
        projection: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        allowed_runner_types: set[str] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        raw: Any = projection.get("candidates", projection) if isinstance(projection, Mapping) else projection
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
            raise OriginalArmStateError("initial executable space must contain candidates")
        required = {"candidate_id", "base_model", "entrypoint", "priority", "runner_type", "status"}
        allowed = {"config_only", "model"} if allowed_runner_types is None else set(allowed_runner_types)
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Mapping):
                raise OriginalArmStateError("initial executable-space entries must be mappings")
            candidate = dict(_jsonable(item))
            missing = sorted(required - set(candidate))
            if missing:
                raise OriginalArmStateError(f"initial candidate missing: {','.join(missing)}")
            candidate_id = str(candidate["candidate_id"])
            if not candidate_id or candidate_id in seen:
                raise OriginalArmStateError(f"duplicate initial candidate: {candidate_id}")
            if str(candidate["runner_type"]) not in allowed:
                raise OriginalArmStateError(f"initial candidate is not executable: {candidate_id}")
            candidate.setdefault("wired", True)
            candidate.setdefault("consumes", [])
            result.append(candidate)
            seen.add(candidate_id)
        return tuple(result)

    @property
    def source_root(self) -> Path:
        return self.source.materialization_root

    @property
    def source_release_digest(self) -> str:
        return self.source.identity_digest

    @property
    def source_file_count(self) -> int:
        return PINNED_ORIGINAL_FILE_COUNT

    @property
    def agent(self) -> Any:
        return self._agent

    def observe(self) -> None:
        """Load native state and mirror the pinned runner's executable boundary."""

        self._agent.observe()
        if not self.use_native_registry:
            return
        for candidate in self._agent.registry:
            if str(candidate.get("runner_type") or "") != "posthoc":
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            self._agent.candidate_health_issues.setdefault(
                candidate_id,
                ["pinned run_candidate.py does not implement posthoc execution"],
            )
            self._agent.quarantined_candidate_ids.add(candidate_id)

    @property
    def arm_id(self) -> str:
        return "original-arm-" + _digest(
            {
                "commit": PINNED_ORIGINAL_COMMIT,
                "source": self.source_release_digest,
                "initial": _digest(self._candidates),
            }
        )[:16]

    @property
    def mutable_paths(self) -> Mapping[str, str]:
        return {
            "campaign_root": str(self.campaign_root),
            "history": str(self._memory.parent),
            "memory": str(self._memory),
            "results": str(self._results),
            "registry": str(self._registry),
            "proposals": str(self._proposals),
            "implementation": str(self.source_root / "recclaw_ext"),
            "checkpoints": str(self._results / "checkpoints"),
        }

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _validate_active_registry(self) -> None:
        try:
            payload = yaml.safe_load(self._registry.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise OriginalArmStateError(
                f"active Original registry cannot be loaded: {self._registry}"
            ) from error
        if not isinstance(payload, Mapping):
            raise OriginalArmStateError(
                f"active Original registry is not a mapping: {self._registry}"
            )
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise OriginalArmStateError(
                f"active Original registry candidates are not a list: {self._registry}"
            )
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in raw_candidates:
            if not isinstance(item, Mapping) or not item.get("candidate_id"):
                raise OriginalArmStateError(
                    f"active Original registry contains an invalid candidate: {self._registry}"
                )
            candidate_id = str(item["candidate_id"])
            if candidate_id in by_id:
                raise OriginalArmStateError(
                    f"active Original registry contains duplicate candidate: {candidate_id}"
                )
            by_id[candidate_id] = item

        missing: list[str] = []
        changed: list[str] = []
        for expected in self._candidates:
            candidate_id = str(expected["candidate_id"])
            actual = by_id.get(candidate_id)
            if actual is None:
                missing.append(candidate_id)
            elif _digest(actual) != _digest(expected):
                changed.append(candidate_id)
        if missing or changed:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if changed:
                details.append("changed=" + ",".join(changed))
            raise OriginalArmStateError(
                "active Original registry failed fixed executable-space validation: "
                + "; ".join(details)
            )

    def _initialize_state(self) -> None:
        self._state.mkdir(parents=True, exist_ok=True)
        manifest = self._state / "arm_manifest.json"
        initial_digest = _digest(self._candidates)
        fresh_campaign = not manifest.exists()
        if not fresh_campaign:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("source_commit") != PINNED_ORIGINAL_COMMIT:
                raise OriginalArmStateError("campaign root has a different pinned commit")
            if existing.get("initial_space_digest") != initial_digest:
                raise OriginalArmStateError("campaign root has a different initial executable space")
            if existing.get("source_release_digest") != self.source_release_digest:
                raise OriginalArmStateError("campaign root has a different pinned source")
            if existing.get("protocol_digest") != _digest(self.protocol):
                raise OriginalArmStateError("campaign root has a different protocol")
            if existing.get("llm_runtime_digest") != _digest(self.llm_runtime):
                raise OriginalArmStateError("campaign root has a different LLM runtime")
        else:
            self._write_json(
                manifest,
                {
                    "schema": "recclaw.original-arm.manifest.v1",
                    "arm_id": self.arm_id,
                    "source_commit": PINNED_ORIGINAL_COMMIT,
                    "source_release_digest": self.source_release_digest,
                    "initial_space_digest": initial_digest,
                    "candidate_count": len(self._candidates),
                    "protocol_digest": _digest(self.protocol),
                    "llm_runtime_digest": _digest(self.llm_runtime),
                    "mutable_paths": self.mutable_paths,
                },
            )
        self._registry.parent.mkdir(parents=True, exist_ok=True)
        if fresh_campaign and not self.use_native_registry:
            # The archive carries the pinned project's historical YAML
            # registry.  The paired arm's active registry is instead the
            # projected fixed-66 state, serialized as JSON (also valid YAML)
            # so the pinned scripts keep their native YAML consumer.
            self._write_json(self._registry, {"candidates": list(self._candidates)})
        self._validate_active_registry()
        self._baseline.mkdir(parents=True, exist_ok=True)
        raw_baselines = self.protocol.get("baseline_results", {})
        if isinstance(raw_baselines, Mapping):
            if "model" in raw_baselines:
                raw_baselines = {str(raw_baselines.get("model")): raw_baselines}
            for model, value in raw_baselines.items():
                if not isinstance(value, Mapping):
                    continue
                record = dict(_jsonable(value))
                record.setdefault("model", str(model))
                record.setdefault("status", "success")
                name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model))
                path = self._baseline / f"{name}.log"
                if not path.exists():
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def _make_agent(self) -> Any:
        proposal_mode = {
            "auto": "algorithm_first",
            "tuning": "conservative",
            "mixed": "mixed",
            "explore": "explore",
        }.get(str(self.loop_mode), "mixed")
        config = self._agent_module.AgentConfig(
            seed=int(self.search_seed),
            proposal_every=max(1, int(self.proposal_every)),
            proposal_count=max(1, int(self.proposal_count)),
            loop_mode=str(self.loop_mode),
            proposal_mode=proposal_mode,
            proposal_source="llm",
            enable_candidate_proposals=True,
            auto_promote_needs_review=True,
            auto_implement_code_required=bool(self.auto_implement_code_required),
            allow_llm_fallback=False,
            memory_read_limit=max(0, int(self.memory_read_limit)),
            prompt_memory_tail=max(0, int(self.prompt_memory_tail)),
            memory_path=self._memory,
            state_summary_path=self._state / "history" / "agent_state_summary.json",
            results_csv=self._results_csv,
            baseline_dir=self._baseline,
            registry_path=self._registry,
            candidate_tree_path=self._results / "candidate_search_tree.json",
            candidate_tree_md_path=self._results / "candidate_search_tree.md",
            candidate_tree_mmd_path=self._results / "candidate_search_tree.mmd",
            experience_summary_path=self._results / "experience_summary.md",
            experience_summary_json_path=self._results / "experience_summary.json",
            reflection_memory_path=self._state / "history" / "reflection_memory.jsonl",
            proposal_path=self._proposals,
            proposal_schema_path=self.source_root / "configs" / "candidate_proposal_schema.yaml",
            checkpoint_dir=str(self._results / "checkpoints"),
            checkpoint_policy=str(self.checkpoint_policy),
            search_intensity=str(self.search_intensity),
            max_pending_implemented=max(1, int(self.max_pending_implemented)),
            max_implement_per_round=max(1, int(self.max_implement_per_round)),
            algorithm_first_explore_rounds=max(
                0, int(self.algorithm_first_explore_rounds)
            ),
            seed_validation_min_metric=float(self.seed_validation_min_metric),
            refresh_experience_every=max(0, int(self.refresh_experience_every)),
            use_experiment_directive=True,
            experiment_directive=str(self.static_experiment_directive),
            static_experiment_directive=str(self.static_experiment_directive),
            llm_provider=str(self.llm_runtime.get("provider", "deepseek")),
            llm_model=str(self.llm_runtime.get("model", "deepseek-chat")),
            llm_base_url=str(self.llm_runtime.get("base_url", "https://api.deepseek.com/v1")),
            llm_api_key_env=str(self.llm_runtime.get("api_key_env", "DEEPSEEK_API_KEY")),
            global_overrides=[str(item) for item in self.protocol.get("global_overrides", ())],
        )
        return self._agent_module.RecClawAgent(config)

    def _install_llm(self) -> None:
        native_responder = self._agent._chat_completion

        def injected(messages: list[dict[str, str]], *, schema_name: str, response_schema: dict[str, Any]) -> str:
            self.llm_call_count += 1
            if self.llm_responder is None:
                response = native_responder(
                    messages,
                    schema_name=schema_name,
                    response_schema=response_schema,
                )
            else:
                response = self.llm_responder(
                    messages,
                    schema_name=schema_name,
                    response_schema=response_schema,
                )
            if isinstance(response, str):
                content = response
            elif isinstance(response, Mapping) and set(response) == {"content"}:
                content = str(response["content"])
            else:
                content = json.dumps(_jsonable(response))
            if schema_name == "recclaw_planner_action":
                content = self._negotiate_planner_action(content)
            return content

        self._agent._chat_completion = injected

    def _negotiate_planner_action(self, content: str) -> str:
        """Keep unsupported native actions out of the one-opportunity round.

        The pinned planner can request seed validation or reporting, but the
        paired Original adapter exposes exactly one physical opportunity per
        round. Negotiate those unsupported actions at the planner boundary so
        the pinned agent still applies its own next-legal-action policy and
        never enters its skip-current-round path.
        """

        try:
            parsed = self._agent._parse_json_loose(content)
        except Exception:
            return content
        payload = self._agent._find_action_payload(parsed)
        if not payload and isinstance(parsed, Mapping):
            payload = dict(parsed)
        requested = str(payload.get("action") or "").strip()
        if requested not in {"multi_seed_verify", "report"}:
            return content

        negotiated = str(self._agent._algorithm_fallback_action()).strip()
        if not negotiated or negotiated in {"multi_seed_verify", "report"}:
            raise OriginalArmStateError(
                "pinned Original has no legal one-opportunity fallback for "
                f"{requested}"
            )
        reason = (
            f"adapter negotiation: unsupported {requested} under the frozen "
            f"one-opportunity budget; routed to native action {negotiated}"
        )
        original_reason = str(payload.get("reason") or "").strip()
        if original_reason:
            reason = f"{reason}; {original_reason}"
        self._last_planner_negotiation = {
            "requested_action": requested,
            "negotiated_action": negotiated,
            "reason": reason,
        }
        updated = dict(payload)
        updated["action"] = negotiated
        updated["reason"] = reason
        return json.dumps(updated, ensure_ascii=True)

    def _round_path(self, round_index: int) -> Path:
        return self._rounds / f"round-{int(round_index):04d}.json"

    def _round_started_path(self, round_index: int) -> Path:
        return self._rounds / f"round-{int(round_index):04d}.started.json"

    def _load_completed(self, round_index: int) -> OriginalArmRoundResultV1 | None:
        path = self._round_path(round_index)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("completed"):
            raise OriginalArmStateError(f"round state is not complete: {path}")
        result = OriginalArmRoundResultV1.from_dict(payload["result"])
        if result.source_release_digest != self.source_release_digest:
            raise OriginalArmStateError(f"round source differs: {path}")
        return result

    def _runner_action(
        self,
        candidate: Mapping[str, Any],
        params: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        run_root = self._results / "candidates" / f"round-{int(round_index):04d}"
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            if self.experiment_runner is None:
                return dict(self._agent.act(dict(candidate), dict(params)))
            raw = self.experiment_runner(dict(candidate), dict(params), int(round_index), run_root)
            if not isinstance(raw, Mapping):
                raise OriginalArmError("experiment runner must return a mapping")
            if isinstance(raw.get("action_out"), Mapping):
                return dict(raw["action_out"])
            if isinstance(raw.get("summary"), Mapping):
                return dict(raw)
            result = dict(_jsonable(raw.get("result", raw)))
            run_id = str(result.get("run_id") or f"original-r{round_index}")
            result["run_id"] = run_id
            result.setdefault("model", candidate.get("base_model", ""))
            result.setdefault("dataset", self.protocol.get("dataset", "ml-1m"))
            result.setdefault("status", result.get("run_status", "missing"))
            result_path = run_root / f"{run_id}.json"
            self._write_json(result_path, result)
            status = str(result["status"]).lower()
            exit_code = int(result.get("exit_code", 0 if status in {"success", "completed", "complete", "finished", "done"} else 1))
            return {
                "exit_code": exit_code,
                "run_status": str(raw.get("run_status") or result["status"]),
                "summary": {
                    "run_id": run_id,
                    "model": result.get("model"),
                    "result_json_path": str(result_path),
                },
            }
        except Exception as error:
            run_id = f"original-crash-r{round_index}"
            result_path = run_root / f"{run_id}.json"
            self._write_json(
                result_path,
                {
                    "run_id": run_id,
                    "model": candidate.get("base_model", ""),
                    "status": "crash",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            return {
                "exit_code": 1,
                "run_status": "CRASH",
                "summary": {
                    "run_id": run_id,
                    "model": candidate.get("base_model", ""),
                    "result_json_path": str(result_path),
                },
            }

    def _persist_result(self, action_out: Mapping[str, Any], result: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
        summary = action_out.get("summary") if isinstance(action_out.get("summary"), Mapping) else {}
        run_id = str(summary.get("run_id") or result.get("run_id") or "")
        if not run_id:
            return ""
        fields = [
            "run_id", "model", "dataset", "config_change", "ndcg@10", "recall@10",
            "mrr@10", "hit@10", "precision@10", "itemcoverage@10", "latency_ms",
            "valid_metric", "run_time", "status", "log_path", "notes",
        ]
        existing: set[str] = set()
        if self._results_csv.exists():
            with self._results_csv.open("r", newline="", encoding="utf-8") as handle:
                existing = {str(row.get("run_id") or "") for row in csv.DictReader(handle)}
        if run_id in existing:
            return run_id
        self._results_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._results_csv.exists() or self._results_csv.stat().st_size == 0
        with self._results_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            row = {
                "run_id": run_id,
                "model": result.get("model") or candidate.get("base_model", ""),
                "dataset": result.get("dataset") or self.protocol.get("dataset", "ml-1m"),
                "config_change": "pinned-original-arm",
                "status": result.get("status") or action_out.get("run_status") or "missing",
                "log_path": summary.get("result_json_path", ""),
                "notes": "pinned_original_arm",
            }
            for field_name in fields:
                row.setdefault(field_name, result.get(field_name, ""))
            writer.writerow(row)
        return run_id

    @staticmethod
    def _outcome(action_out: Mapping[str, Any], result: Mapping[str, Any]) -> str:
        statuses = {
            str(result.get("status") or "").upper(),
            str(action_out.get("run_status") or "").upper(),
        }
        if statuses & {"CENSORED", "CENSOR"}:
            return OriginalOutcomeClass.CENSORED.value
        if statuses & {"MISSING", "OUTCOME_MISSING", "INCOMPLETE", "NO_RESULT", "NOT_AVAILABLE"}:
            return OriginalOutcomeClass.OUTCOME_MISSING.value
        try:
            exit_code = int(action_out.get("exit_code", 1))
        except (TypeError, ValueError):
            exit_code = 1
        if exit_code or str(result.get("status") or "").lower() not in {"success", "completed", "complete", "finished", "done"}:
            return OriginalOutcomeClass.CRASH.value
        return OriginalOutcomeClass.SUCCESS.value

    def _projection(self, proposal_count: int) -> dict[str, Any]:
        return {
            "memory_count": len(self._agent.memory),
            "registry_count": len(self._agent.registry),
            "proposal_count": proposal_count,
            "proposal_ids": sorted(str(item.get("candidate_id") or "") for item in self._agent.candidate_proposals),
            "history_candidate_ids": sorted(str(key) for key in self._agent.history_by_candidate),
        }

    def _missing_round(
        self,
        round_index: int,
        *,
        reason: str = "pinned planner skipped the configured opportunity",
        outcome_class: str = OriginalOutcomeClass.OUTCOME_MISSING.value,
    ) -> OriginalArmRoundResultV1:
        run_id = f"original-missing-r{round_index}"
        self._agent.remember_event(
            {
                "event": "round_opportunity_missing",
                "round_id": round_index,
                "run_id": run_id,
                "outcome_class": outcome_class,
                "reason": reason,
            }
        )
        proposal_count = len(self._agent.candidate_proposals)
        return OriginalArmRoundResultV1(
            arm_id=self.arm_id,
            round_index=round_index,
            resumed=False,
            execution_opportunities=1,
            execution_calls=0,
            selected_candidate={},
            selected_params={},
            candidate_id="",
            run_id=run_id,
            outcome_class=outcome_class,
            candidate_result={"status": "missing"},
            compare_baseline={"decision": "missing", "delta": None},
            compare_history_best={"decision": "missing", "delta": None},
            dimension_report={},
            decision="missing",
            reason=reason,
            next_action="resume the next configured Original opportunity",
            proposal_count=proposal_count,
            memory_digest=_digest(self._agent.memory),
            source_release_digest=self.source_release_digest,
            state_projection=self._projection(proposal_count),
            state_paths=dict(self.mutable_paths),
        )

    def record_missing_round(
        self,
        round_index: int,
        *,
        reason: str = "external scheduler censored the configured opportunity",
        censored: bool = True,
    ) -> OriginalArmRoundResultV1:
        """Advance arm-local memory after an opportunity cannot be replayed."""

        round_index = int(round_index)
        if round_index < 1:
            raise OriginalArmStateError("round_index must be positive")
        completed = self._load_completed(round_index)
        if completed is not None:
            return completed.resumed_result()
        result = self._missing_round(
            round_index,
            reason=reason,
            outcome_class=(
                OriginalOutcomeClass.CENSORED.value
                if censored
                else OriginalOutcomeClass.OUTCOME_MISSING.value
            ),
        )
        self._write_json(
            self._round_path(round_index),
            {"completed": True, "result": result.to_dict()},
        )
        return result

    def run_round(self, round_index: int, *, force_proposal_refresh: bool = False) -> OriginalArmRoundResultV1:
        """Run the pinned order once; a completed round is never executed twice."""

        round_index = int(round_index)
        if round_index < 1:
            raise OriginalArmStateError("round_index must be positive")
        completed = self._load_completed(round_index)
        if completed is not None:
            return completed.resumed_result()
        started_path = self._round_started_path(round_index)
        if started_path.is_file():
            return self.record_missing_round(
                round_index,
                reason="Original opportunity was interrupted before a sealed result",
                censored=True,
            )
        self._write_json(
            started_path,
            {
                "schema": "recclaw.original-arm.round-started.v1",
                "round_index": round_index,
                "source_release_digest": self.source_release_digest,
                "protocol_digest": _digest(self.protocol),
            },
        )

        self.observe()
        if round_index == 1:
            self._agent._refresh_experience_artifacts(0, reason="initial")
        self._agent.remember_experiment_directive()
        self._agent.reset_round_policy()
        self._last_planner_negotiation = None
        self._agent.apply_auto_planner(round_index)
        if self._last_planner_negotiation is not None:
            self._agent.remember_event(
                {
                    "event": "planner_action_negotiated",
                    "round_id": round_index,
                    **self._last_planner_negotiation,
                }
            )
            self._last_planner_negotiation = None
        if self._agent.skip_current_round:
            result = self._missing_round(round_index)
            self._write_json(self._round_path(round_index), {"completed": True, "result": result.to_dict()})
            return result
        self._agent.force_proposal_refresh = bool(force_proposal_refresh) or self._agent.force_proposal_refresh
        self._agent.refresh_candidate_proposals(round_index)
        proposal_count = len(self._agent.candidate_proposals)
        candidate, params, _context = self._agent.plan()
        candidate_id = str(candidate.get("candidate_id") or "")
        self._agent.scheduled_candidate_ids.add(candidate_id)
        parameter_signature = str(candidate.get("parameter_signature") or "")
        if parameter_signature:
            self._agent.scheduled_param_signatures.add(parameter_signature)
        execution_signature = str(candidate.get("execution_signature") or self._agent._execution_signature(candidate, params))
        if execution_signature:
            self._agent.scheduled_execution_signatures.add(execution_signature)

        action_out = self._runner_action(candidate, params, round_index)
        candidate_result, compare_baseline, compare_history, dimension_report = self._agent.evaluate(candidate, action_out)
        run_id = self._persist_result(action_out, candidate_result, candidate)
        decision, reason, next_action = self._agent.reflect(
            candidate, params, action_out, candidate_result, compare_baseline, compare_history
        )
        self._agent.retain_checkpoints_for_result(decision=decision, run_id=run_id, seed_validation={})
        record = self._agent_module.TrialRecord(
            round_id=round_index,
            candidate_id=candidate_id,
            params=dict(params),
            run_id=run_id,
            status=str(candidate_result.get("status") or "unknown"),
            result=dict(candidate_result),
            compare_baseline=dict(compare_baseline),
            compare_history_best=dict(compare_history),
            dimension_report=dict(dimension_report),
            decision=decision,
            reason=reason,
            next_action=next_action,
            parent_candidate_id=str(candidate.get("parent_candidate_id") or ""),
            proposal_id=str(candidate.get("proposal_id") or ""),
            parameter_signature=parameter_signature,
            execution_signature=execution_signature,
            proposal_source=str(candidate.get("proposal_source") or ""),
            is_baseline_improvement=float(compare_baseline.get("delta") or 0.0) > self._agent.config.min_keep_delta
            if compare_baseline.get("delta") is not None
            else False,
            is_history_best=float(compare_history.get("delta") or 0.0) > self._agent.config.min_keep_delta
            if compare_history.get("delta") is not None
            else False,
        )
        self._agent.remember(record)
        self._agent._update_history_best(candidate.get("base_model") or candidate_result.get("model"), candidate_result)
        if (
            self._agent.config.refresh_experience_every > 0
            and round_index % self._agent.config.refresh_experience_every == 0
        ):
            self._agent._refresh_experience_artifacts(
                round_index, reason=f"round_{round_index}"
            )
        result = OriginalArmRoundResultV1(
            arm_id=self.arm_id,
            round_index=round_index,
            resumed=False,
            execution_opportunities=1,
            execution_calls=1,
            selected_candidate=dict(candidate),
            selected_params=dict(params),
            candidate_id=candidate_id,
            run_id=run_id,
            outcome_class=self._outcome(action_out, candidate_result),
            candidate_result=dict(candidate_result),
            compare_baseline=dict(compare_baseline),
            compare_history_best=dict(compare_history),
            dimension_report=dict(dimension_report),
            decision=str(decision),
            reason=str(reason),
            next_action=str(next_action),
            proposal_count=proposal_count,
            memory_digest=_digest(self._agent.memory),
            source_release_digest=self.source_release_digest,
            state_projection=self._projection(proposal_count),
            state_paths=dict(self.mutable_paths),
        )
        self._write_json(self._round_path(round_index), {"completed": True, "result": result.to_dict()})
        return result


OriginalRecClawArmV1 = PinnedOriginalArmV1
OriginalArmV1 = PinnedOriginalArmV1


__all__ = [
    "PINNED_ORIGINAL_COMMIT",
    "PINNED_ORIGINAL_FILE_COUNT",
    "FINAL_SELF_REFLECTION_DIRECTIVE",
    "OriginalArmError",
    "OriginalArmSourceError",
    "OriginalArmStateError",
    "OriginalOutcomeClass",
    "PinnedOriginalSourceV1",
    "OriginalArmRoundResultV1",
    "PinnedOriginalArmV1",
    "OriginalRecClawArmV1",
    "OriginalArmV1",
]
