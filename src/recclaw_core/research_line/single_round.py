"""Production composition for one physical Research Line round.

The module freezes one pre-outcome manifest, then invokes the unified runtime.
It deliberately has no paired-arm logic and no alternate Research controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sqlite3
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    ExperimentBindingV1,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
    MAX_REPAIR_TURNS,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    load_lab_api_credential_pairs,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    meta_v20_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchExecutableProfileV1,
    SearchProfileActivationV1,
    bind_search_candidate,
    freeze_experiment_slate,
    route_frozen_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_development_validation_profile_manifest,
    campaign_development_validation_profile_ref,
)
from recclaw_core.experiments.helix_abc_v1.realization_identity import (
    bl_icf_search_space_conformance,
)
from recclaw_core.search_spaces.bl_icf_v1 import PROVIDER as BL_ICF_PROVIDER
from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    run_disposable_fixed_batch_resource_probe,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityKindV1,
)

from .fresh_runner import Launcher, make_fresh_runner
from .interfaces import ResearchContext
from .provider import (
    ProviderCall,
    ProviderImplementerGateway,
    ProviderResearchProducer,
    RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
)
from .replay import OfflineProducerReplayV1
from .search_space_adapter import SearchSpaceAdapter
from .runtime import (
    InnovationRuntimeInputs,
    MetaResearchInputs,
    activate_promoted_meta_strategy,
    activate_staged_innovation,
    bindings_for_context,
    resolver_environment_for_profile,
    run_research_round,
)


DEFAULT_DISCOVERY_TRAINING_SEED = 54201

_IMPLEMENTATION_REQUIREMENTS = (
    "RecBole GeneralRecommender interface",
    "candidate-local recclaw_ext package",
    "finite pairwise loss and full-sort scores",
)
_COMPATIBILITY_REQUIREMENTS = (
    "general collaborative filtering",
    "offline top-n evaluation",
    "pairwise input",
    "train-only fitting",
)
_DEVELOPMENT_SEARCH_FILES = frozenset(
    {
        "ml-1m.train.inter",
        "ml-1m.dev.inter",
        "ml-1m.user",
        "ml-1m.item",
    }
)


def _bytes_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_search_partition_identity(
    verified_assets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the active development-only search data before launch.

    Legacy BL-ICF callers keep the campaign development profile below.  A
    declarative single-parent profile instead supplies the canonical asset
    identity already produced by its READY loader, so the shared runtime does
    not reinterpret every dataset as the BL-ICF four-file layout.
    """

    if verified_assets is not None:
        if not isinstance(verified_assets, Mapping):
            raise ValueError("verified search-data identity must be a mapping")
        search_data = verified_assets.get("search_data")
        assets = verified_assets.get("assets")
        if (
            verified_assets.get("schema") != "recclaw.execution-assets.v1"
            or not isinstance(search_data, Mapping)
            or not isinstance(assets, (list, tuple))
        ):
            raise ValueError("verified search-data identity is incomplete")
        root_value = search_data.get("data_path")
        manifest_value = search_data.get("manifest_ref")
        manifest_sha256 = search_data.get("manifest_sha256")
        if not isinstance(root_value, str) or not isinstance(manifest_value, str):
            raise ValueError("verified search-data paths are invalid")
        root = Path(root_value).resolve()
        manifest_path = Path(manifest_value).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as error:
            raise ValueError("search-data manifest escapes the active root") from error
        if not root.is_dir() or not manifest_path.is_file():
            raise ValueError("verified search-data root or manifest is missing")
        actual_manifest_hash = _bytes_sha256(manifest_path)
        if actual_manifest_hash != manifest_sha256:
            raise ValueError("verified search-data manifest hash mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("verified search-data manifest is unreadable") from error
        dataset = manifest.get("dataset") if isinstance(manifest, Mapping) else None
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("verified search-data manifest lacks dataset identity")
        dataset_root = (root / dataset).resolve()
        if not dataset_root.is_dir():
            raise ValueError("verified search-data dataset root is missing")
        if (dataset_root / f"{dataset}.heldout.inter").exists():
            raise ValueError("selected development search root exposes heldout data")
        declared_hashes: dict[str, str] = {}
        search_roles: set[str] = set()
        for row in assets:
            if not isinstance(row, Mapping):
                raise ValueError("verified search-data asset row is invalid")
            role = row.get("role")
            path_value = row.get("path")
            declared_hash = row.get("sha256")
            if not isinstance(role, str) or not isinstance(path_value, str):
                raise ValueError("verified search-data asset identity is invalid")
            path = Path(path_value).resolve()
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if (
                not isinstance(declared_hash, str)
                or len(declared_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in declared_hash
                )
            ):
                raise ValueError(f"search-data asset hash is invalid: {role}")
            if not path.is_file() or _bytes_sha256(path) != declared_hash:
                raise ValueError(f"search-data asset hash mismatch: {role}")
            declared_hashes[relative_path] = declared_hash
            search_roles.add(role)
        required_roles = {
            "SEARCH_DATA_MANIFEST",
            "SEARCH_TRAIN_INTERACTIONS",
            "SEARCH_DEVELOPMENT_INTERACTIONS",
        }
        if not required_roles.issubset(search_roles):
            raise ValueError("verified search-data identity lacks train/dev assets")
        manifest_relative = manifest_path.relative_to(root).as_posix()
        if declared_hashes.get(manifest_relative) != actual_manifest_hash:
            raise ValueError("verified search-data manifest asset is not bound")
        return canonical_value(
            {
                "dataset": dataset,
                "dataset_root": str(dataset_root),
                "root": str(root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": actual_manifest_hash,
                "file_hashes": declared_hashes,
            }
        )

    manifest_path = fresh_r1.SEARCH_DATA_ROOT / "search_partition_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"selected search partition manifest is missing: {manifest_path}")
    development_profile = campaign_development_validation_profile_manifest()
    dataset = development_profile["dataset"]
    actual_manifest_hash = _bytes_sha256(manifest_path)
    files = dataset["file_sha256"]
    if not isinstance(files, Mapping) or set(files) != _DEVELOPMENT_SEARCH_FILES:
        raise ValueError(
            "development profile must bind exactly train, dev, user, and item files"
        )
    dataset_root = (fresh_r1.SEARCH_DATA_ROOT / "ml-1m").resolve()
    if (dataset_root / "ml-1m.heldout.inter").exists():
        raise ValueError("selected development search root exposes heldout data")
    declared_hashes: dict[str, str] = {}
    for filename in sorted(_DEVELOPMENT_SEARCH_FILES):
        declared_hash = files[filename]
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdef" for character in declared_hash)
        ):
            raise ValueError(f"search partition manifest hash is invalid: {filename}")
        file_path = (dataset_root / filename).resolve()
        try:
            file_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"search partition manifest file escapes the ml-1m root: {filename}"
            ) from error
        if not file_path.is_file():
            raise ValueError(
                f"search partition manifest file is missing: {filename}"
            )
        actual_hash = _bytes_sha256(file_path)
        if actual_hash != declared_hash:
            raise ValueError(
                f"search partition manifest file hash mismatch: {filename}"
            )
        declared_hashes[filename] = declared_hash
    return canonical_value(
        {
            "manifest_sha256": actual_manifest_hash,
            "file_hashes": declared_hashes,
        }
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_identity(repo_root: Path) -> dict[str, Any]:
    roots = (
        repo_root / "src/recclaw_core/research_line",
        repo_root / "src/recclaw_core/experiments/helix_abc_v1",
        repo_root / "recclaw_ext",
        repo_root / "configs",
        repo_root / "scripts/campaign_train_worker.py",
        repo_root / "scripts/run_research_line_single_round.py",
        repo_root
        / "tests/experiments/helix_abc_v1/fixtures/innovation_spine",
    )
    rows = []
    for root in roots:
        paths = (root,) if root.is_file() else root.rglob("*")
        for path in sorted(paths):
            if not path.is_file() or path.suffix not in {
                ".py",
                ".json",
                ".txt",
                ".yaml",
                ".inter",
            }:
                continue
            rows.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "sha256": _bytes_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return canonical_value(
        {
            "file_count": len(rows),
            "source_tree_digest": sha256_digest(rows),
        }
    )


def _git_head(repo_root: Path, *, fallback_env: str = "RECCLAW_SOURCE_HEAD") -> str:
    try:
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        delegated = os.environ.get(fallback_env, "").strip()
        if len(delegated) != 40 or any(
            character not in "0123456789abcdef" for character in delegated
        ):
            raise ValueError(
                f"source checkout has no Git identity and {fallback_env} is invalid"
            ) from None
        return delegated


@dataclass(frozen=True, slots=True)
class ResearchBaselineSourceV1:
    """Research-owned baseline source, supplied by the caller explicitly.

    Use :meth:`from_receipt_path` for the legacy receipt contract or
    :meth:`from_identity` when a standalone Research campaign already owns a
    content-addressed baseline observation.
    """

    source_ref: str
    source_sha256: str | None = None
    receipt_path: Path | None = None
    comparator_ref: str | None = None
    comparator_digest: str | None = None
    frozen_ndcg_at_10: float | None = None
    protocol_digest: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ValueError("Research baseline source_ref must be non-empty")
        object.__setattr__(self, "source_ref", self.source_ref.strip())
        if self.receipt_path is not None:
            if not isinstance(self.receipt_path, Path):
                raise ValueError("Research baseline receipt_path must be a Path")
            object.__setattr__(self, "receipt_path", self.receipt_path.resolve())
            if any(
                value is not None
                for value in (
                    self.comparator_ref,
                    self.comparator_digest,
                    self.frozen_ndcg_at_10,
                    self.protocol_digest,
                    self.seed,
                )
            ):
                raise ValueError(
                    "receipt-backed Research baseline must not mix identity fields"
                )
            if self.source_sha256 is not None and not _is_sha256(
                self.source_sha256
            ):
                raise ValueError("Research baseline source_sha256 is invalid")
            return
        if not _is_sha256(self.source_sha256):
            raise ValueError(
                "identity-backed Research baseline requires source_sha256"
            )
        if (
            not isinstance(self.comparator_ref, str)
            or not self.comparator_ref.strip()
            or not isinstance(self.comparator_digest, str)
            or not _is_sha256(self.comparator_digest)
        ):
            raise ValueError("identity-backed Research baseline comparator identity is invalid")
        if (
            isinstance(self.frozen_ndcg_at_10, bool)
            or not isinstance(self.frozen_ndcg_at_10, (int, float))
            or not math.isfinite(float(self.frozen_ndcg_at_10))
        ):
            raise ValueError("identity-backed Research baseline NDCG@10 is invalid")
        if not _is_sha256(self.protocol_digest):
            raise ValueError("identity-backed Research baseline protocol_digest is invalid")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("identity-backed Research baseline seed is invalid")
        object.__setattr__(self, "comparator_ref", self.comparator_ref.strip())
        object.__setattr__(
            self,
            "frozen_ndcg_at_10",
            float(self.frozen_ndcg_at_10),
        )

    @classmethod
    def from_receipt_path(cls, path: Path) -> "ResearchBaselineSourceV1":
        resolved = path.resolve()
        return cls(
            source_ref=f"receipt:{resolved.as_posix()}",
            receipt_path=resolved,
        )

    @classmethod
    def from_identity(
        cls,
        *,
        source_ref: str,
        source_sha256: str,
        comparator_ref: str,
        comparator_digest: str,
        frozen_ndcg_at_10: float,
        protocol_digest: str,
        seed: int,
    ) -> "ResearchBaselineSourceV1":
        return cls(
            source_ref=source_ref,
            source_sha256=source_sha256,
            comparator_ref=comparator_ref,
            comparator_digest=comparator_digest,
            frozen_ndcg_at_10=frozen_ndcg_at_10,
            protocol_digest=protocol_digest,
            seed=seed,
        )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": "recclaw.research-line.baseline-source.v1",
                "kind": "RECEIPT" if self.receipt_path is not None else "IDENTITY",
                "source_ref": self.source_ref,
                "source_sha256": self.source_sha256,
                "receipt_path": (
                    str(self.receipt_path) if self.receipt_path is not None else None
                ),
                "comparator_ref": self.comparator_ref,
                "comparator_digest": self.comparator_digest,
                "frozen_ndcg_at_10": self.frozen_ndcg_at_10,
                "protocol_digest": self.protocol_digest,
                "seed": self.seed,
            }
        )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _resolve_baseline_source(
    *,
    research_baseline_source: ResearchBaselineSourceV1 | None,
    incumbent_receipt_path: Path | None,
) -> ResearchBaselineSourceV1:
    if research_baseline_source is not None and incumbent_receipt_path is not None:
        raise ValueError(
            "pass either research_baseline_source or incumbent_receipt_path, not both"
        )
    if research_baseline_source is not None:
        if not isinstance(research_baseline_source, ResearchBaselineSourceV1):
            raise ValueError(
                "research_baseline_source must be ResearchBaselineSourceV1"
            )
        return research_baseline_source
    if incumbent_receipt_path is not None:
        return ResearchBaselineSourceV1.from_receipt_path(incumbent_receipt_path)
    raise ValueError(
        "explicit Research baseline source is required; pass "
        "research_baseline_source or incumbent_receipt_path"
    )


def _incumbent(
    source: ResearchBaselineSourceV1,
    *,
    protocol_digest: str,
    seed: int,
) -> dict[str, Any]:
    if source.receipt_path is None:
        if source.protocol_digest != protocol_digest or source.seed != seed:
            raise ValueError(
                "Research baseline identity does not match the selected protocol or seed"
            )
        return canonical_value(
            {
                "comparator_ref": source.comparator_ref,
                "comparator_digest": source.comparator_digest,
                "frozen_ndcg@10": source.frozen_ndcg_at_10,
                "source_ref": source.source_ref,
                "source_receipt": None,
                "source_receipt_sha256": source.source_sha256,
            }
        )
    path = source.receipt_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload.get("typed_episode", {})
    summary = payload.get("outcome_summary", {})
    baseline = summary.get("baseline_metrics", {})
    if (
        episode.get("protocol_digest") != protocol_digest
        or int(summary.get("seed", -1)) != seed
        or not isinstance(baseline.get("ndcg@10"), (int, float))
    ):
        raise ValueError("incumbent observation does not match protocol, seed, and NDCG@10")
    source_sha256 = _bytes_sha256(path)
    if source.source_sha256 is not None and source.source_sha256 != source_sha256:
        raise ValueError("Research baseline receipt source_sha256 does not match")
    return canonical_value(
        {
            "comparator_ref": episode["comparator_ref"],
            "comparator_digest": episode["comparator_digest"],
            "frozen_ndcg@10": float(baseline["ndcg@10"]),
            "source_ref": source.source_ref,
            "source_receipt": str(path),
            "source_receipt_sha256": source_sha256,
        }
    )


def _context(
    *,
    repo_root: Path,
    campaign_id: str,
    profile: SearchExecutableProfileV1,
    policy: VersionedResearchPolicyV1,
    incumbent: Mapping[str, Any],
) -> ResearchContext:
    knowledge_source = repo_root / "docs/search_spaces/bl_icf_mechanism_space_v1.md"
    return ResearchContext(
        campaign_id=campaign_id,
        round_index=1,
        knowledge_base={
            "kind": "RECOMMENDER_METHOD_KNOWLEDGE_BASE",
            "bl_icf_source": "repo:docs/search_spaces/bl_icf_mechanism_space_v1.md",
            "bl_icf_source_sha256": _bytes_sha256(knowledge_source),
            "active_executable_capability_count": len(profile.entries),
            "open_capability_escape_hatch": True,
            "protocol": (
                "ML-1M train/development-validation full-sort with "
                "best-validation feedback; heldout is post-selection only"
            ),
        },
        frozen_goal={
            "metric": "NDCG@10",
            "direction": "maximize",
            "research_objective": (
                "improve the recommendation frontier while learning which "
                "mechanism caused the observed signal"
            ),
            "ordinary_experiment_opportunities": 1,
            "single_seed_claim_ceiling": "INCONCLUSIVE",
        },
        frontier={
            "incumbent_ndcg@10": incumbent["frozen_ndcg@10"],
            "incumbent_ref": incumbent["comparator_ref"],
            "incumbent_digest": incumbent["comparator_digest"],
        },
        scientific_memory={
            "by_role": {role: {} for role in DISCOVERY_PRODUCERS},
            "prior_round_count": 0,
            "negative_evidence": (),
        },
        unresolved_questions=(
            {
                "question": (
                    "Which executable mechanism can improve NDCG@10 beyond "
                    "the frozen incumbent under the same protocol?"
                ),
                "mechanism_axis": "architecture",
            },
            {
                "question": (
                    "What competing explanation would reproduce the selected "
                    "mechanism's predicted signature?"
                ),
                "mechanism_axis": "objective",
            },
        ),
        policy=policy.to_dict(),
        budget={
            "producer_logical_calls": 4,
            "producer_token_ceiling_each": (
                RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
            ),
            "implementer_logical_calls_max": MAX_REPAIR_TURNS + 1,
            "implementer_token_ceiling_each": fresh_r1.IMPLEMENTATION_TOKEN_CEILING,
            "physical_attempts_per_logical_call_max": fresh_r1.MAX_PHYSICAL_ATTEMPTS,
            "experiment_opportunities": 1,
        },
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
    )


@dataclass(frozen=True, slots=True)
class SingleRoundComposition:
    repo_root: Path
    run_root: Path
    api_config_path: Path
    incumbent_receipt_path: Path | None
    baseline_source: ResearchBaselineSourceV1
    seed: int
    epochs: int
    timeout_seconds: int
    watchdog_seconds: int
    profile: SearchExecutableProfileV1
    policy: VersionedResearchPolicyV1
    context: ResearchContext
    incumbent: Mapping[str, Any]
    recbole_identity: Mapping[str, Any]
    manifest: Mapping[str, Any]
    search_space_adapter: SearchSpaceAdapter | None = None


def compose_single_round(
    *,
    repo_root: Path,
    run_root: Path,
    api_config_path: Path,
    campaign_id: str,
    seed: int = DEFAULT_DISCOVERY_TRAINING_SEED,
    epochs: int = fresh_r1.EXPERIMENT_EPOCHS,
    timeout_seconds: int = 1800,
    watchdog_seconds: int = 1800,
    incumbent_receipt_path: Path | None = None,
    research_baseline_source: ResearchBaselineSourceV1 | None = None,
    search_space_adapter: SearchSpaceAdapter | None = None,
) -> SingleRoundComposition:
    """Resolve immutable inputs without a physical call.

    ``research_baseline_source`` is the standalone Research contract.  The
    explicit ``incumbent_receipt_path`` remains a compatibility alias; no
    arm-labeled Research receipt is selected implicitly.
    """

    repo_root = repo_root.resolve()
    run_root = run_root.resolve()
    api_config_path = api_config_path.resolve()
    baseline_source = _resolve_baseline_source(
        research_baseline_source=research_baseline_source,
        incumbent_receipt_path=incumbent_receipt_path,
    )
    development_profile = campaign_development_validation_profile_manifest()
    development_protocol = development_profile["protocol"]
    strict_protocol_ref = str(development_protocol["protocol_ref"])
    strict_protocol_digest = str(development_protocol["protocol_digest"])
    if (
        baseline_source.receipt_path is None
        and baseline_source.protocol_digest != strict_protocol_digest
    ):
        raise ValueError(
            "single-round baseline protocol is outside strict BL-ICF"
        )
    conformance = bl_icf_search_space_conformance(
        space_identity=BL_ICF_PROVIDER.identity().to_dict(),
        profile_ref=campaign_development_validation_profile_ref(),
        fixed_fallback=False,
    )
    if (
        conformance["ordered_primitive_ids_count"] != 264
        or conformance["fixed_fallback"] is not False
    ):
        raise ValueError("strict BL-ICF search-space conformance drift")
    profile = SearchExecutableProfileV1(
        campaign_id=campaign_id,
        profile_ref=str(conformance["search_space_id"]),
        profile_digest=str(conformance["search_space_digest"]),
        protocol_ref=strict_protocol_ref,
        protocol_digest=strict_protocol_digest,
        activation=SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN,
        predecessor_campaign_id=None,
        predecessor_profile_ref=None,
        predecessor_profile_digest=None,
        entries=(),
    )
    policy = meta_v20_research_control_policy()
    incumbent = _incumbent(
        baseline_source,
        protocol_digest=profile.protocol_digest,
        seed=seed,
    )
    context = _context(
        repo_root=repo_root,
        campaign_id=campaign_id,
        profile=profile,
        policy=policy,
        incumbent=incumbent,
    )
    credential_pairs = load_lab_api_credential_pairs(api_config_path)
    endpoint_digests = tuple(
        sha256_digest({"base_url": base_url.rstrip("/")})
        for base_url, _api_key in credential_pairs
    )
    recbole_identity = fresh_r1.recbole_source_identity(fresh_r1.RECBole_ROOT)
    dataset_identity = validated_search_partition_identity()
    proposal_prompt = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_open_spec_proposal_prompt_v1.txt"
    )
    proposal_schema = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_open_spec_proposal_response_v1.schema.json"
    )
    implementer_prompt = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_implementer_prompt_v1.txt"
    )
    implementer_schema = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/fresh_r1_implementation_response_v1.schema.json"
    )
    manifest = canonical_value(
        {
            "schema": "recclaw.research-line.single-round-pre-outcome.v1",
            "campaign_id": campaign_id,
            "round_index": 1,
            "run_root": str(run_root),
            "source": {
                "git_head": _git_head(repo_root),
                **_source_identity(repo_root),
            },
            "context": {
                "ref": context.context_ref,
                "digest": context.digest,
                "value": context.to_dict(),
            },
            "active_profile": {
                "ref": profile.profile_ref,
                "digest": profile.profile_digest,
                "entry_count": len(profile.entries),
                "entries": tuple(entry.canonical_dict() for entry in profile.entries),
            },
            "policy": {"digest": policy.digest, "value": policy.to_dict()},
            "provider": {
                "model": fresh_r1.MODEL,
                "config_sha256": _bytes_sha256(api_config_path),
                "credential_count": len(credential_pairs),
                "ordered_endpoint_digests": endpoint_digests,
                "proposal_prompt_sha256": _bytes_sha256(proposal_prompt),
                "proposal_schema_sha256": _bytes_sha256(proposal_schema),
                "implementer_prompt_sha256": _bytes_sha256(implementer_prompt),
                "implementer_schema_sha256": _bytes_sha256(implementer_schema),
                "call_budget": context.budget,
            },
            "experiment": {
                "dataset": "ml-1m",
                "dataset_manifest_sha256": dataset_identity["manifest_sha256"],
                "dataset_file_hashes": dataset_identity["file_hashes"],
                "split": canonical_value(DEVELOPMENT_SPLIT),
                "evaluator": dict(DEVELOPMENT_EVALUATOR),
                "metric_contract_digest": sha256_digest(DEVELOPMENT_EVALUATOR),
                "seed": seed,
                "epochs": epochs,
                "timeout_seconds": timeout_seconds,
                "watchdog_seconds": watchdog_seconds,
                "experiment_opportunities": 1,
                "incumbent": incumbent,
                "baseline_source": baseline_source.canonical_dict(),
            },
            "runtime": {
                "python_executable": str(fresh_r1.PYTHON_EXECUTABLE.resolve()),
                "python_sha256": _bytes_sha256(fresh_r1.PYTHON_EXECUTABLE),
                "python_version": platform.python_version(),
                "recbole_commit": _git_head(
                    fresh_r1.RECBole_ROOT,
                    fallback_env="RECCLAW_RECBOLE_HEAD",
                ),
                "recbole_source_identity": recbole_identity,
            },
            "claim_boundary": {
                "development_only": True,
                "online_partition_role": "DEVELOPMENT_VALIDATION",
                "heldout_access": "POST_SELECTION_ONLY",
                "metric_source": "BEST_VALID_RESULT",
                "single_seed_scientific_claim": False,
                "qualification_is_outcome_evidence": False,
            },
        }
    )
    return SingleRoundComposition(
        repo_root=repo_root,
        run_root=run_root,
        api_config_path=api_config_path,
        incumbent_receipt_path=baseline_source.receipt_path,
        baseline_source=baseline_source,
        seed=seed,
        epochs=epochs,
        timeout_seconds=timeout_seconds,
        watchdog_seconds=watchdog_seconds,
        profile=profile,
        policy=policy,
        context=context,
        incumbent=incumbent,
        recbole_identity=recbole_identity,
        manifest=manifest,
        search_space_adapter=search_space_adapter,
    )


class _RecordedProducer:
    """Call Provider once per role; replay the same drafts in offline Meta."""

    def __init__(self, provider: ProviderResearchProducer) -> None:
        self.provider = provider
        self._records: dict[
            str,
            CandidateProposalV4 | Mapping[str, Any] | Exception,
        ] = {}

    @property
    def call_traces(self) -> tuple[Mapping[str, Any], ...]:
        return self.provider.call_traces

    def __call__(
        self,
        role: str,
        view: Mapping[str, Any],
    ) -> CandidateProposalV4 | Mapping[str, Any]:
        record = self._records.get(role)
        if record is None:
            try:
                provider_result = self.provider(role, view)
                record = (
                    provider_result
                    if isinstance(provider_result, CandidateProposalV4)
                    else canonical_value(dict(provider_result))
                )
            except Exception as error:
                self._records[role] = error
                raise
            self._records[role] = record
        if isinstance(record, Exception):
            raise RuntimeError(f"recorded Producer failure: {type(record).__name__}")
        return record


class _RecordedExperimentRunner:
    """Persist the physical observation before downstream interpretation."""

    def __init__(self, runner: Any, output_path: Path) -> None:
        self.runner = runner
        self.output_path = output_path

    def __call__(self, recipe: Mapping[str, Any], binding: Any) -> Mapping[str, Any]:
        result = canonical_value(dict(self.runner(recipe, binding)))
        _write_new_json(self.output_path, result)
        return result


class _RecordedFailureProducer:
    """Replay persisted Producer failures without another Provider call."""

    def __init__(self, traces: tuple[Mapping[str, Any], ...]) -> None:
        self._by_role = {
            str(trace["logical_call_id"]).split(":")[-2]: trace for trace in traces
        }
        self._emitted: list[Mapping[str, Any]] = []

    @property
    def call_traces(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._emitted)

    def __call__(self, role: str, _view: Mapping[str, Any]) -> Mapping[str, Any]:
        trace = self._by_role[role]
        if trace not in self._emitted:
            self._emitted.append(trace)
        failure = trace.get("failure")
        detail = failure if isinstance(failure, Mapping) else {}
        raise RuntimeError(
            "recorded Provider failure: "
            + str(detail.get("failure_class", "PROVIDER_ERROR"))
        )


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _recover_candidate_run(composition: SingleRoundComposition) -> Mapping[str, Any]:
    experiment_root = composition.run_root / "execution/experiments/round-01"
    binding_payload = _read_json(experiment_root / "experiment_binding.json")
    binding = ExperimentBindingV1.from_canonical_dict(binding_payload)
    recipe = _read_json(experiment_root / "execution_recipe.json")
    worker_path = experiment_root / "worker/worker_result.json"
    worker = _read_json(worker_path)
    telemetry_path = experiment_root / "worker/resource_telemetry.json"
    telemetry = _read_json(telemetry_path)
    recbole_identity = _read_json(experiment_root / "recbole_source_identity.json")
    if canonical_value(recipe) != canonical_value(binding.worker_recipe()):
        raise ValueError("persisted worker recipe differs from Experiment Binding")
    if (
        binding.seed != composition.seed
        or binding.epochs != composition.epochs
        or binding.execution_purpose
        != "RESEARCH_LINE_SINGLE_ROUND_OFFLINE_TOPN"
    ):
        raise ValueError("persisted Experiment Binding differs from frozen round inputs")
    if worker.get("model") != binding.model:
        raise ValueError("persisted worker model differs from Experiment Binding")
    if (
        binding.split != DEVELOPMENT_SPLIT
        or canonical_value(dict(binding.evaluator)) != DEVELOPMENT_EVALUATOR
    ):
        raise ValueError("persisted Experiment Binding is not development-only")
    metrics, metric_identity_matches = fresh_r1._round_feedback_metrics(
        worker,
        DEVELOPMENT_EVALUATOR,
    )
    if not metric_identity_matches:
        raise ValueError("persisted worker metric identity is not development-only")
    phase_records = telemetry.get("phase_records", ())
    if not isinstance(phase_records, (tuple, list)):
        raise ValueError("persisted resource telemetry lacks phase records")
    wall_time_ms = int(telemetry.get("initialization_wall_time_ms", 0)) + sum(
        int(record.get("wall_time_ms", 0))
        for record in phase_records
        if isinstance(record, Mapping)
    )
    if wall_time_ms < 1:
        raise ValueError("persisted resource telemetry lacks positive wall time")
    log_path = experiment_root / "worker/work/log/training.log"
    return canonical_value(
        {
            "binding_digest": binding.digest,
            "experiment_binding": binding.canonical_dict(),
            "experiment_binding_digest": binding.digest,
            "experiment_binding_ref": binding.ref,
            "execution_recipe_digest": binding.execution_recipe_digest,
            "exit_status": worker.get("exit_status"),
            "filesystem_mount_audit": worker.get("filesystem_mount_audit"),
            "metric_source": worker.get("metric_source"),
            "metrics": metrics,
            "model": worker.get("model"),
            "online_partition_role": worker.get("online_partition_role"),
            "recbole_source_identity": recbole_identity,
            "resource_telemetry": telemetry,
            "resource_telemetry_sha256": _bytes_sha256(telemetry_path),
            "result_sha256": _bytes_sha256(worker_path),
            "log_sha256": _bytes_sha256(log_path) if log_path.is_file() else None,
            "seed": binding.seed,
            "wall_time_ms": wall_time_ms,
            "recovery_projection": {
                "physical_execution_reused": True,
                "new_physical_experiment_calls": 0,
                "source_round_ref": (
                    "ROUND_FAILURE.json"
                    if (composition.run_root / "ROUND_FAILURE.json").is_file()
                    else "ROUND_01_TRACE.json"
                ),
                "wall_time_source": "DURABLE_RESOURCE_TELEMETRY_SUM",
            },
        }
    )


def _persisted_provider_replay(
    source_run_root: Path,
) -> tuple[ProviderCall, list[str]]:
    source_trace = _read_json(source_run_root / "ROUND_01_TRACE.json")
    traces = source_trace.get("provider_traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("source round lacks persisted Provider traces")
    trace_by_id = {
        str(trace["logical_call_id"]): trace
        for trace in traces
        if isinstance(trace, Mapping)
    }
    calls: dict[str, fresh_r1.ProviderAttemptResult] = {}
    for database in sorted((source_run_root / "provider_calls").rglob("broker.sqlite3")):
        connection = sqlite3.connect(database)
        try:
            row = connection.execute(
                "SELECT logical_call_id, request_digest, response_digest, "
                "response_json, input_tokens, cached_input_tokens, output_tokens, "
                "total_tokens, latency_ms, returned_model FROM calls "
                "WHERE status = 'SUCCESS'"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            continue
        logical_call_id = str(row[0])
        trace = trace_by_id.get(logical_call_id)
        if trace is None:
            raise ValueError("Provider broker call is absent from the source trace")
        response = json.loads(str(row[3]))
        if not isinstance(response, Mapping):
            raise ValueError("persisted Provider response is not an object")
        calls[logical_call_id] = fresh_r1.ProviderAttemptResult(
            call=fresh_r1.CanaryBrokerCallV1(
                logical_call_id=logical_call_id,
                request_digest=str(row[1]),
                response_digest=str(row[2]),
                response=canonical_value(response),
                input_tokens=int(row[4]),
                cached_input_tokens=int(row[5]),
                output_tokens=int(row[6]),
                total_tokens=int(row[7]),
                latency_ms=int(row[8]),
                returned_model=str(row[9]),
            ),
            attempts=tuple(trace.get("attempts", ())),
            failure=(
                trace.get("failure")
                if isinstance(trace.get("failure"), Mapping)
                else None
            ),
        )
    if set(calls) != set(trace_by_id):
        raise ValueError("Provider broker successes do not match the source trace")
    emitted: list[str] = []

    def replay(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        logical_call_id = str(kwargs.get("logical_call_id", ""))
        if logical_call_id not in calls:
            raise ValueError("recovery requested an unrecorded Provider call")
        if logical_call_id in emitted:
            raise ValueError("recovery attempted to replay one logical call twice")
        emitted.append(logical_call_id)
        return calls[logical_call_id]

    return replay, emitted


def execute_single_round(
    composition: SingleRoundComposition,
    *,
    provider_call: ProviderCall | None = None,
    launch: Launcher | None = None,
) -> Mapping[str, Any]:
    """Write the manifest first, then run one physical experiment opportunity."""

    if composition.run_root.exists():
        raise FileExistsError("single-round run_root already exists")
    composition.run_root.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        composition.run_root / "PRE_OUTCOME_MANIFEST.json",
        composition.manifest,
    )
    call_root = composition.run_root / "provider_calls"
    provider = ProviderResearchProducer(
        config_source=composition.api_config_path,
        call_root=call_root,
        session_id=composition.context.campaign_id,
        provider_call=provider_call,
    )
    producer = _RecordedProducer(provider)
    implementer = ProviderImplementerGateway(
        config_source=composition.api_config_path,
        call_root=call_root,
        session_id=composition.context.campaign_id,
        provider_call=provider_call,
    )
    implementation_prompt = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_implementer_prompt_v1.txt"
    )
    implementer_policy = fresh_r1._shared_policy(
        _bytes_sha256(implementation_prompt),
        sha256_digest(
            {
                "tools": (),
                "network": False,
                "allowed_files": (
                    "recclaw_ext/__init__.py",
                    "recclaw_ext/candidate.py",
                    "recclaw_ext/trainer.py",
                ),
            }
        ),
        allowed_files=(
            "recclaw_ext/__init__.py",
            "recclaw_ext/candidate.py",
            "recclaw_ext/trainer.py",
        ),
        execution_contract=None,
    )

    def fixture_factory(policy: Any, attempt: int, candidate_root: Path) -> Any:
        contract = policy.execution_contract
        if not isinstance(contract, Mapping):
            raise ValueError("qualified OpenSpec lacks execution_contract")
        fixture = fresh_r1._qualification_fixture(
            composition.repo_root,
            seed=composition.seed,
            root=(
                composition.run_root
                / "innovation_qualification"
                / candidate_root.parent.name
            ),
            base_model_config=str(contract["base_model_config"]),
        )
        return replace(
            fixture,
            runtime_identity_ref=policy.runtime_identity_ref,
            runtime_identity_digest=policy.runtime_identity_digest,
        )

    def unit_check_factory(policy: Any) -> Any:
        contract = policy.execution_contract
        if not isinstance(contract, Mapping):
            raise ValueError("qualified OpenSpec lacks execution_contract")
        return fresh_r1._shared_behavioral_unit_check(
            {},
            base_model_config=str(contract["base_model_config"]),
        )

    innovation = InnovationRuntimeInputs(
        implementer=implementer,
        policy=implementer_policy,
        candidate_parent=composition.run_root / "innovation_candidates",
        fixture_factory=fixture_factory,
        unit_check_factory=unit_check_factory,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="single-round-qualified-v1",
        registry_version="single-round-registry-v1",
        predecessor_registry_ref=composition.profile.profile_ref,
        predecessor_registry_digest=composition.profile.profile_digest,
        profile_version="single-round-next-fresh-profile-v1",
        fresh_campaign_id=composition.context.campaign_id + ":next-fresh",
        resource_admission_required=True,
        resource_probe=lambda **kwargs: run_disposable_fixed_batch_resource_probe(
            composition.repo_root,
            total_budget_seconds=fresh_r1.MAX_WORKER_CEILING_SECONDS,
            **kwargs,
        ),
        resource_probe_parent=composition.run_root / "resource_probes",
    )
    producer_bindings = bindings_for_context(
        composition.context,
        active_profile=composition.profile,
        implementation_requirements=_IMPLEMENTATION_REQUIREMENTS,
        compatibility_requirements=_COMPATIBILITY_REQUIREMENTS,
    )
    resolver_environment = resolver_environment_for_profile(
        composition.profile,
        available_dependencies=fresh_r1.AVAILABLE_DEPENDENCIES,
        budget_limits=fresh_r1.BUDGET_LIMITS,
        protocol_requirements=fresh_r1.PROTOCOL_REQUIREMENTS,
    )
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=4,
    )
    replay = OfflineProducerReplayV1(
        producer=producer,
        producer_bindings=producer_bindings,
        equal_replay_token_charge=RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
        deterministic_directive_replay=True,
    )
    runner = _RecordedExperimentRunner(
        make_fresh_runner(
            repo_root=composition.repo_root,
            side_root=composition.run_root / "execution",
            run_id="round-01",
            seed=composition.seed,
            epochs=composition.epochs,
            timeout_seconds=composition.timeout_seconds,
            execution_purpose="RESEARCH_LINE_SINGLE_ROUND_OFFLINE_TOPN",
            candidate_root_by_capability={},
            recbole_commit_identity=str(
                composition.manifest["runtime"]["recbole_commit"]
            ),
            expected_recbole_source_tree_digest=str(
                composition.recbole_identity["source_tree_digest"]
            ),
            resource_telemetry=True,
            watchdog_seconds=composition.watchdog_seconds,
            launch=launch,
        ),
        composition.run_root / "PHYSICAL_EXPERIMENT_OBSERVATION.json",
    )
    try:
        result = run_research_round(
            context=composition.context,
            active_profile=composition.profile,
            producer=producer,
            producer_bindings=producer_bindings,
            resolver_environment=resolver_environment,
            carryover_proposals=(),
            budget_snapshot={"experiment_opportunities": 1},
            router=router,
            policy=composition.policy,
            memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
            runner=runner,
            incumbent_observation=composition.incumbent,
            metric_contract_digest=sha256_digest(DEVELOPMENT_EVALUATOR),
            evaluator=DEVELOPMENT_EVALUATOR,
            split=DEVELOPMENT_SPLIT,
            frozen_profile_ref=campaign_development_validation_profile_ref(),
            observation_seed=str(composition.seed),
            next_discriminative_test=(
                "Use the Episode to schedule the most discriminative matched "
                "control, ablation, or seed only when scientifically warranted."
            ),
            innovation_inputs=innovation,
            meta_research_inputs=MetaResearchInputs(
                offline_replay=replay,
                next_campaign_id=composition.context.campaign_id + ":next-fresh",
            ),
            search_space_adapter=composition.search_space_adapter,
        )
    except Exception as error:
        _write_new_json(
            composition.run_root / "ROUND_FAILURE.json",
            {
                "schema": "recclaw.research-line.single-round-failure.v1",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "provider_traces": producer.call_traces,
                "mechanism_negative_evidence": False,
            },
        )
        raise

    _write_new_json(composition.run_root / "ROUND_01_TRACE.json", result.to_dict())
    round_two: Mapping[str, Any] | None = None
    if result.innovation is not None and result.innovation.activation_ready:
        next_profile, successor, candidate, qualified_execution = (
            activate_staged_innovation(result)
        )
        if result.meta_research is not None and result.meta_research.activated_policy:
            successor, successor_policy = activate_promoted_meta_strategy(
                result,
                next_profile=next_profile,
            )
        else:
            successor_policy = result.interpretation.policy_successor
        binding = bind_search_candidate(
            profile=next_profile,
            proposal=candidate,
            capability_ref=candidate.capability_ref,
        )
        slate = freeze_experiment_slate(
            profile=next_profile,
            bindings=(binding,),
            budget_snapshot={"experiment_opportunities": 1},
        )
        acquisition = route_frozen_experiment_slate(
            profile=next_profile,
            slate=slate,
            router=router,
            policy_projection=successor_policy.to_dict(),
        )
        round_two = canonical_value(
            {
                "schema": "recclaw.research-line.round-two-preexecution.v1",
                "context": successor.to_dict(),
                "context_ref": successor.context_ref,
                "context_digest": successor.digest,
                "producer_inputs_digest": successor.producer_inputs_digest,
                "active_profile_ref": next_profile.profile_ref,
                "active_profile_digest": next_profile.profile_digest,
                "active_profile_entry_count": len(next_profile.entries),
                "activated_candidate": candidate.to_dict(),
                "qualified_execution": qualified_execution,
                "search_binding": binding.canonical_dict(),
                "search_binding_digest": binding.digest,
                "acquisition": {
                    "slate_ref": acquisition.slate_ref,
                    "slate_digest": acquisition.slate_digest,
                    "route_trace": acquisition.route_trace.to_dict(),
                    "route_trace_digest": acquisition.route_trace.digest,
                    "decisions": tuple(
                        decision.to_dict() for decision in acquisition.decisions
                    ),
                    "selected_binding": (
                        acquisition.selected_binding.canonical_dict()
                        if acquisition.selected_binding is not None
                        else None
                    ),
                },
                "selected_candidate_id": (
                    acquisition.selected_binding.proposal.candidate_id
                    if acquisition.selected_binding is not None
                    else None
                ),
                "behavior_changed_fields": result.interpretation.behavior_before.changed_fields(
                    result.interpretation.behavior_after
                ),
                "experiment_executed": False,
            }
        )
        _write_new_json(
            composition.run_root / "ROUND_02_PREEXECUTION.json",
            round_two,
        )
    complete = bool(
        result.candidate_run is not None
        and result.candidate_run.get("exit_status") == "SUCCESS"
        and result.interpretation is not None
        and getattr(result.interpretation, "episode", None) is not None
        and result.meta_research is not None
        and result.innovation is not None
        and result.innovation.activation_ready
        and round_two is not None
        and round_two["active_profile_entry_count"] == 67
        and round_two["selected_candidate_id"]
        == round_two["activated_candidate"]["candidate_id"]
    )
    summary = canonical_value(
        {
            "schema": "recclaw.research-line.single-round-summary.v1",
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "context_digest": composition.context.digest,
            "producer_outcome_count": len(result.producer_outcomes),
            "provider_trace_count": len(result.provider_traces),
            "experiment_executed": result.candidate_run is not None,
            "experiment_exit_status": (
                result.candidate_run.get("exit_status")
                if result.candidate_run is not None
                else None
            ),
            "typed_episode_created": bool(
                result.interpretation is not None
                and getattr(result.interpretation, "episode", None) is not None
            ),
            "innovation_admitted": bool(
                result.innovation is not None and result.innovation.admitted
            ),
            "next_fresh_search_consumed": round_two is not None,
            "meta_shadow_evaluated": result.meta_research is not None,
            "round_two_behavior_changed_fields": (
                round_two["behavior_changed_fields"] if round_two else ()
            ),
        }
    )
    _write_new_json(composition.run_root / "ROUND_SUMMARY.json", summary)
    return summary


def recover_single_round(composition: SingleRoundComposition) -> Mapping[str, Any]:
    """Close feedback from a completed physical run after a downstream crash."""

    if not composition.run_root.is_dir():
        raise FileNotFoundError("single-round run_root does not exist")
    manifest_path = composition.run_root / "PRE_OUTCOME_MANIFEST.json"
    failure_path = composition.run_root / "ROUND_FAILURE.json"
    original_manifest = _read_json(manifest_path)
    failure = _read_json(failure_path)
    original_context = original_manifest.get("context")
    if (
        not isinstance(original_context, Mapping)
        or not isinstance(original_context.get("value"), Mapping)
    ):
        raise ValueError("recovery requires the complete pre-outcome Context")
    context_payload = dict(original_context["value"])
    context_payload.pop("schema", None)
    context = ResearchContext(**context_payload)
    if context.digest != original_context.get("digest"):
        raise ValueError("pre-outcome Context digest does not match its value")
    original_policy = original_manifest.get("policy")
    if (
        not isinstance(original_policy, Mapping)
        or not isinstance(original_policy.get("value"), Mapping)
    ):
        raise ValueError("recovery requires the complete pre-outcome policy")
    policy_payload = dict(original_policy["value"])
    for field_name in (
        "producer_token_allocation",
        "mechanism_axis_targeting",
        "router_priors",
        "acquisition_parameters",
    ):
        policy_payload[field_name] = tuple(
            tuple(item) if isinstance(item, (tuple, list)) else item
            for item in policy_payload[field_name]
        )
    policy = VersionedResearchPolicyV1(**policy_payload)
    if (
        policy.digest != original_policy.get("digest")
        or context.policy != policy.to_dict()
    ):
        raise ValueError("pre-outcome policy digest does not match its value")
    if (
        context.active_profile_ref != composition.profile.profile_ref
        or context.active_profile_digest != composition.profile.profile_digest
    ):
        raise ValueError("recovery profile differs from the pre-outcome manifest")
    traces = tuple(failure.get("provider_traces", ()))
    if len(traces) != len(DISCOVERY_PRODUCERS) or not all(
        isinstance(trace, Mapping) for trace in traces
    ):
        raise ValueError("recovery requires the persisted four-Producer trace set")
    producer = _RecordedFailureProducer(traces)
    producer_bindings = bindings_for_context(
        context,
        active_profile=composition.profile,
        implementation_requirements=_IMPLEMENTATION_REQUIREMENTS,
        compatibility_requirements=_COMPATIBILITY_REQUIREMENTS,
    )
    resolver_environment = resolver_environment_for_profile(
        composition.profile,
        available_dependencies=fresh_r1.AVAILABLE_DEPENDENCIES,
        budget_limits=fresh_r1.BUDGET_LIMITS,
        protocol_requirements=fresh_r1.PROTOCOL_REQUIREMENTS,
    )
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=4,
    )
    replay = (
        OfflineProducerReplayV1(
            producer=producer,
            producer_bindings=producer_bindings,
            equal_replay_token_charge=RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
            deterministic_directive_replay=True,
        )
        if policy.meta_router_policy_digest is not None
        else None
    )
    candidate_run = _recover_candidate_run(composition)
    runner_calls = 0

    def recovered_runner(_recipe: Mapping[str, Any], _binding: Any) -> Mapping[str, Any]:
        nonlocal runner_calls
        runner_calls += 1
        return candidate_run

    result = run_research_round(
        context=context,
        active_profile=composition.profile,
        producer=producer,
        producer_bindings=producer_bindings,
        resolver_environment=resolver_environment,
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=router,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=recovered_runner,
        incumbent_observation=composition.incumbent,
        metric_contract_digest=sha256_digest(DEVELOPMENT_EVALUATOR),
        evaluator=DEVELOPMENT_EVALUATOR,
        split=DEVELOPMENT_SPLIT,
        frozen_profile_ref=campaign_development_validation_profile_ref(),
        observation_seed=str(composition.seed),
        next_discriminative_test=(
            "Use the Episode to schedule the most discriminative matched "
            "control, ablation, or seed only when scientifically warranted."
        ),
        meta_research_inputs=(
            MetaResearchInputs(
                offline_replay=replay,
                next_campaign_id=context.campaign_id + ":recovered-next",
            )
            if replay is not None
            else None
        ),
        search_space_adapter=composition.search_space_adapter,
    )
    if runner_calls != 1 or result.interpretation is None:
        raise ValueError("recovery did not consume exactly one persisted observation")
    trace = result.to_dict()
    _write_new_json(
        composition.run_root / "ROUND_01_TRACE_RECOVERED.json",
        trace,
    )
    successor = result.interpretation.successor_context
    _write_new_json(
        composition.run_root / "ROUND_02_SUCCESSOR_RECOVERED.json",
        {
            "schema": "recclaw.research-line.recovered-successor.v1",
            "context": successor.to_dict(),
            "context_ref": successor.context_ref,
            "context_digest": successor.digest,
            "policy": result.interpretation.policy_successor.to_dict(),
            "behavior_changed_fields": result.interpretation.behavior_before.changed_fields(
                result.interpretation.behavior_after
            ),
            "next_discriminative_task": result.interpretation.next_discriminative_task.to_dict(),
        },
    )
    summary = canonical_value(
        {
            "schema": "recclaw.research-line.single-round-recovery.v1",
            "status": "RECOVERED_EPISODE_INCOMPLETE_ARCHITECTURE",
            "original_manifest_sha256": _bytes_sha256(manifest_path),
            "original_failure_sha256": _bytes_sha256(failure_path),
            "original_failure_type": failure.get("error_type"),
            "provider_trace_count": len(traces),
            "new_provider_calls": 0,
            "new_physical_experiment_calls": 0,
            "physical_observation_reused": True,
            "experiment_exit_status": result.candidate_run.get("exit_status"),
            "typed_episode_created": result.interpretation.episode is not None,
            "mechanism_attribution": result.interpretation.mechanism_attribution,
            "comparator_delta": result.interpretation.search_utility_event.comparator_delta,
            "innovation_admitted": False,
            "meta_shadow_evaluated": result.meta_research is not None,
            "successor_context_ref": successor.context_ref,
            "successor_context_digest": successor.digest,
        }
    )
    _write_new_json(
        composition.run_root / "ROUND_RECOVERY_SUMMARY.json",
        summary,
    )
    return summary


def recover_incomplete_single_round(
    source: SingleRoundComposition,
    recovery: SingleRoundComposition,
) -> Mapping[str, Any]:
    """Replay sealed pre-outcome calls after a generic qualifier-binding fix."""

    if source.run_root == recovery.run_root:
        raise ValueError("recovery must use a fresh non-overwriting run root")
    source_manifest_path = source.run_root / "PRE_OUTCOME_MANIFEST.json"
    source_trace_path = source.run_root / "ROUND_01_TRACE.json"
    source_summary_path = source.run_root / "ROUND_SUMMARY.json"
    source_manifest = _read_json(source_manifest_path)
    source_trace = _read_json(source_trace_path)
    source_summary = _read_json(source_summary_path)
    if (
        source_summary.get("status") != "INCOMPLETE"
        or source_summary.get("typed_episode_created") is not True
        or source_summary.get("experiment_exit_status") != "SUCCESS"
        or source_summary.get("innovation_admitted") is not False
    ):
        raise ValueError("source round is not the recoverable incomplete P8 shape")
    source_context = source_manifest.get("context")
    if (
        not isinstance(source_context, Mapping)
        or source_context.get("digest") != recovery.context.digest
        or source.context.digest != recovery.context.digest
        or source.profile.profile_digest != recovery.profile.profile_digest
        or source.policy.digest != recovery.policy.digest
        or source.seed != recovery.seed
        or source.epochs != recovery.epochs
        or canonical_value(source.incumbent) != canonical_value(recovery.incumbent)
    ):
        raise ValueError("recovery changed a frozen scientific input")
    innovation = source_trace.get("innovation")
    attempts = innovation.get("attempts") if isinstance(innovation, Mapping) else None
    if not isinstance(attempts, list) or not attempts or any(
        not isinstance(attempt, Mapping)
        or not isinstance(attempt.get("failure"), Mapping)
        or attempt["failure"].get("reason_code")
        != "PACKAGE_RUNTIME_BINDING_MISMATCH"
        for attempt in attempts
    ):
        raise ValueError("source Innovation failure is not the fixed runtime binding bug")

    provider_replay, replayed_call_ids = _persisted_provider_replay(source.run_root)
    persisted_candidate_run = _recover_candidate_run(source)
    launch_count = 0

    def replay_launch(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal launch_count
        launch_count += 1
        if launch_count != 1:
            raise ValueError("recovery attempted more than one experiment replay")
        return persisted_candidate_run

    ordinary_summary = execute_single_round(
        recovery,
        provider_call=provider_replay,
        launch=replay_launch,
    )
    recovered_trace = _read_json(recovery.run_root / "ROUND_01_TRACE.json")
    recovered_provider_traces = recovered_trace.get("provider_traces")
    source_provider_traces = source_trace.get("provider_traces")
    if not isinstance(recovered_provider_traces, list) or not isinstance(
        source_provider_traces, list
    ):
        raise ValueError("recovery lacks comparable Provider traces")
    source_provider_by_id = {
        str(trace["logical_call_id"]): trace
        for trace in source_provider_traces
        if isinstance(trace, Mapping)
    }
    for trace in recovered_provider_traces:
        if not isinstance(trace, Mapping):
            raise ValueError("recovered Provider trace is invalid")
        original = source_provider_by_id.get(str(trace.get("logical_call_id")))
        if original is None or any(
            trace.get(field_name) != original.get(field_name)
            for field_name in ("kind", "logical_call_id", "prompt_digest")
        ):
            raise ValueError("recovered Provider prompt identity changed")
    round_two = _read_json(recovery.run_root / "ROUND_02_PREEXECUTION.json")
    recovered_innovation = recovered_trace.get("innovation")
    recovered_qualification = (
        recovered_innovation.get("qualification")
        if isinstance(recovered_innovation, Mapping)
        else None
    )
    recovered_receipt = (
        recovered_qualification.get("receipt")
        if isinstance(recovered_qualification, Mapping)
        else None
    )
    if (
        ordinary_summary.get("status") != "COMPLETE"
        or launch_count != 1
        or not isinstance(recovered_receipt, Mapping)
        or recovered_receipt.get("status") != "PASS"
        or int(round_two.get("active_profile_entry_count", 0)) != 67
        or round_two.get("selected_candidate_id")
        != round_two.get("activated_candidate", {}).get("candidate_id")
    ):
        raise ValueError("recovery did not close Innovation into next-fresh Search")
    provenance = canonical_value(
        {
            "schema": "recclaw.research-line.incomplete-round-recovery.v1",
            "status": "COMPLETE",
            "completion_kind": "SEALED_PREOUTCOME_REPLAY_AFTER_GENERIC_QUALIFIER_FIX",
            "source_run_root": str(source.run_root),
            "source_manifest_sha256": _bytes_sha256(source_manifest_path),
            "source_trace_sha256": _bytes_sha256(source_trace_path),
            "source_summary_sha256": _bytes_sha256(source_summary_path),
            "source_provider_trace_count": len(source_provider_traces),
            "replayed_logical_call_ids": tuple(replayed_call_ids),
            "replayed_provider_trace_count": len(recovered_provider_traces),
            "new_provider_calls": 0,
            "physical_observation_reused": True,
            "new_physical_experiment_calls": 0,
            "source_physical_observation_sha256": _bytes_sha256(
                source.run_root / "PHYSICAL_EXPERIMENT_OBSERVATION.json"
            ),
            "typed_episode_created": ordinary_summary["typed_episode_created"],
            "innovation_admitted": ordinary_summary["innovation_admitted"],
            "next_fresh_search_consumed": ordinary_summary[
                "next_fresh_search_consumed"
            ],
            "meta_shadow_evaluated": ordinary_summary["meta_shadow_evaluated"],
            "round_two_behavior_changed_fields": ordinary_summary[
                "round_two_behavior_changed_fields"
            ],
            "qualification_status": recovered_receipt["status"],
            "qualification_stage": recovered_receipt["stage"],
            "active_profile_entry_count": round_two["active_profile_entry_count"],
            "selected_candidate_id": round_two["selected_candidate_id"],
        }
    )
    _write_new_json(
        recovery.run_root / "ROUND_RECOVERY_PROVENANCE.json",
        provenance,
    )
    return provenance


__all__ = [
    "ResearchBaselineSourceV1",
    "SingleRoundComposition",
    "compose_single_round",
    "execute_single_round",
    "recover_incomplete_single_round",
    "recover_single_round",
    "validated_search_partition_identity",
]
