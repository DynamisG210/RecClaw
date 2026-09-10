"""Small shared bridge from one editable source file to the E1 RecBole ABI.

The bridge is intentionally controller-neutral.  It materializes the exact
three-file ``recclaw_ext`` package, runs mechanical qualification before any
training debit, and delegates metric-bearing work to the existing trusted
development runner.  It never accepts or discovers an outer-heldout input.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalQualificationRun,
    MechanicalRecBoleAdapterV1,
    candidate_tree_identity,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CandidatePackageV1,
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    QualificationStatusV1,
    RealizationModeV1,
)
from recclaw_core.experiments.helix_abc_v1.model_configuration import (
    MODEL_CONFIG_MAPPING_REQUIREMENT,
    bind_model_configuration_source,
)


E1_BRIDGE_ID = "recclaw.e1.native-multvae-bridge.v1"
E1_PARENT_ID = "e1-multvae-pilot-rh-parent-v1"
E1_ENTRYPOINT = "recclaw_ext.models.e1_multvae:FreshCandidateModel"
E1_MODEL = "MultiVAE"
E1_RECBOLE_COMMIT = "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
E1_CAPABILITY_FAMILY = "E1_MULTVAE_GENERAL_RECOMMENDER"
E1_ALLOWED_FILES = (
    "recclaw_ext/__init__.py",
    "recclaw_ext/models/__init__.py",
    "recclaw_ext/models/e1_multvae.py",
)
E1_MODEL_RELATIVE_PATH = E1_ALLOWED_FILES[-1]
E1_TRAINING_CONFIG = canonical_value(
    {
        # RecBole's MultiVAE ABI treats this as the combined mu/logvar width.
        # 256 therefore preserves the reviewed 128-dimensional latent z.
        "latent_dimension": 256,
        "mlp_hidden_size": (384,),
        "dropout_prob": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 2000,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "train_batch_size": 256,
        "stopping_step": 5,
        # RecBole's built-in ML-1M default loads only interaction columns.
        # Load the search-visible item identity column explicitly so the
        # training and final-only evaluator share the complete 3883-item
        # catalog vocabulary without consulting heldout interactions.
        "load_col": {
            "inter": ("user_id", "item_id"),
            "item": ("item_id",),
        },
        "metric_decimal_place": 8,
        "valid_metric": "NDCG@10",
        "valid_metric_bigger": True,
        "topk": (10,),
    }
)
E1_SEARCH_FILE_SHA256 = canonical_value(
    {
        "ml-1m/ml-1m.train.inter": (
            "c84b1a4f6c6d974f32f126b173f11f7af8e12e1a143a50ac5a53e9945903491a"
        ),
        "ml-1m/ml-1m.dev.inter": (
            "631911b8e59d312e110ba7205151bcda3d52378ecbf510d48d8b9cc162956cc9"
        ),
        "ml-1m/ml-1m.user": (
            "bbc7b4f8d3204d00e0465cf1451b8e8bae2c9e94378292223603c229cf2df09f"
        ),
        "ml-1m/ml-1m.item": (
            "6eff097333db47adf003039eb90cb6ab13978fa9577d915d2366e4d073844dc3"
        ),
    }
)
E1_PROTOCOL = canonical_value(
    {
        "dataset": COMMON_DATASET,
        "files": E1_SEARCH_FILE_SHA256,
        "fit_partition": "SEARCH_TRAIN",
        "feedback_partition": "SEARCH_DEVELOPMENT",
        "candidate_universe": "FULL_SORT",
        "metric": "NDCG@10",
        "outer_heldout_access": "FORBIDDEN_DURING_SEARCH",
        "training_seed": 54201,
    }
)
E1_PROTOCOL_REF = "recclaw.e1.ml1m.train-dev-fullsort.v1"
E1_PROTOCOL_DIGEST = sha256_digest(E1_PROTOCOL)

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_PARENT_PACKAGE_ROOT = _REPO_ROOT / "e1_native_parent"
_DEFAULT_SEARCH_MANIFEST = _HERE.parent / "resources/e1_search_manifest.v1.json"


class E1NativeBridgeError(RuntimeError):
    """The candidate cannot cross the fixed E1 native bridge."""


@dataclass(frozen=True, slots=True)
class E1NativeCandidate:
    candidate_root: Path
    entrypoint: str
    source_sha256: str
    source_tree_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    file_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_root": str(self.candidate_root),
                "entrypoint": self.entrypoint,
                "source_sha256": self.source_sha256,
                "model_configuration_abi": MODEL_CONFIG_MAPPING_REQUIREMENT,
                "entrypoint_source_sha256": self.file_sha256[E1_MODEL_RELATIVE_PATH],
                "source_tree_digest": self.source_tree_digest,
                "candidate_root_ref": self.candidate_root_ref,
                "candidate_root_digest": self.candidate_root_digest,
                "file_sha256": self.file_sha256,
            }
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_source(payload: bytes) -> str:
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise E1NativeBridgeError("candidate source must be UTF-8") from error
    if not source or "\x00" in source:
        raise E1NativeBridgeError("candidate source is empty or contains NUL")
    try:
        tree = ast.parse(source, filename=E1_MODEL_RELATIVE_PATH, mode="exec")
        compile(tree, E1_MODEL_RELATIVE_PATH, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise E1NativeBridgeError("candidate source is not valid Python") from error
    classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    if "FreshCandidateModel" not in classes:
        raise E1NativeBridgeError(
            "candidate source must define top-level FreshCandidateModel"
        )
    return source


def parent_package_identity() -> Mapping[str, Any]:
    """Return the exact three-file frozen parent identity."""

    manifest = snapshot_candidate_tree(_PARENT_PACKAGE_ROOT)
    paths = tuple(row["path"] for row in manifest)
    if paths != E1_ALLOWED_FILES:
        raise E1NativeBridgeError("frozen E1 parent file set drift")
    source_tree_digest = sha256_digest({"files": manifest})
    return canonical_value(
        {
            "bridge_id": E1_BRIDGE_ID,
            "parent_id": E1_PARENT_ID,
            "entrypoint": E1_ENTRYPOINT,
            "allowed_files": E1_ALLOWED_FILES,
            "file_sha256": {
                row["path"]: row["sha256"] for row in manifest
            },
            "source_tree_digest": source_tree_digest,
        }
    )


def materialize_candidate_source(
    *,
    source_path: Path,
    candidate_root: Path,
    expected_source_sha256: str | None = None,
) -> E1NativeCandidate:
    """Copy one controller-owned class source into the fixed three-file ABI."""

    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise E1NativeBridgeError("candidate source does not exist")
    source_payload = source_path.read_bytes()
    source = _model_source(source_payload)
    source_sha256 = _sha256(source_payload)
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise E1NativeBridgeError("candidate source SHA-256 mismatch")
    bound_source_payload = bind_model_configuration_source(source).encode("utf-8")

    root = Path(candidate_root).absolute()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise E1NativeBridgeError("candidate root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    for relative in E1_ALLOWED_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            bound_source_payload
            if relative == E1_MODEL_RELATIVE_PATH
            else (_PARENT_PACKAGE_ROOT / relative).read_bytes()
        )
        with target.open("xb") as handle:
            handle.write(payload)

    manifest = snapshot_candidate_tree(root)
    if tuple(row["path"] for row in manifest) != E1_ALLOWED_FILES:
        raise E1NativeBridgeError("materialized candidate file set drift")
    candidate_root_ref = (
        "e1-native-candidate-root:"
        + sha256_digest(
            {
                "bridge_id": E1_BRIDGE_ID,
                "source_sha256": source_sha256,
            }
        )
    )
    source_tree_digest, candidate_root_digest = candidate_tree_identity(
        root,
        candidate_root_ref=candidate_root_ref,
    )
    return E1NativeCandidate(
        candidate_root=root.resolve(),
        entrypoint=E1_ENTRYPOINT,
        source_sha256=source_sha256,
        source_tree_digest=source_tree_digest,
        candidate_root_ref=candidate_root_ref,
        candidate_root_digest=candidate_root_digest,
        file_sha256=canonical_value(
            {row["path"]: row["sha256"] for row in manifest}
        ),
    )


def _load_materialized_candidate(
    *,
    candidate_root: Path,
    expected_source_payload: bytes,
) -> E1NativeCandidate:
    """Recover only an exact bridge-owned materialization for result replay."""

    root = Path(candidate_root).resolve()
    bound_source_payload = bind_model_configuration_source(
        _model_source(expected_source_payload)
    ).encode("utf-8")
    manifest = snapshot_candidate_tree(root)
    if tuple(row["path"] for row in manifest) != E1_ALLOWED_FILES:
        raise E1NativeBridgeError("existing materialized candidate file set drift")
    for relative in E1_ALLOWED_FILES:
        expected = (
            bound_source_payload
            if relative == E1_MODEL_RELATIVE_PATH
            else (_PARENT_PACKAGE_ROOT / relative).read_bytes()
        )
        if (root / relative).read_bytes() != expected:
            raise E1NativeBridgeError(
                f"existing materialized candidate bytes drift: {relative}"
            )
    source_sha256 = _sha256(expected_source_payload)
    candidate_root_ref = (
        "e1-native-candidate-root:"
        + sha256_digest(
            {
                "bridge_id": E1_BRIDGE_ID,
                "source_sha256": source_sha256,
            }
        )
    )
    source_tree_digest, candidate_root_digest = candidate_tree_identity(
        root,
        candidate_root_ref=candidate_root_ref,
    )
    return E1NativeCandidate(
        candidate_root=root,
        entrypoint=E1_ENTRYPOINT,
        source_sha256=source_sha256,
        source_tree_digest=source_tree_digest,
        candidate_root_ref=candidate_root_ref,
        candidate_root_digest=candidate_root_digest,
        file_sha256=canonical_value(
            {row["path"]: row["sha256"] for row in manifest}
        ),
    )


def verify_search_manifest(
    *,
    manifest_path: Path = _DEFAULT_SEARCH_MANIFEST,
    data_root: Path | None = None,
) -> Mapping[str, Any]:
    """Verify only the four search-visible files; never enumerate a parent root."""

    path = Path(manifest_path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E1NativeBridgeError("E1 search manifest is unavailable") from error
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != "e1-ml1m-v1"
        or document.get("dataset") != COMMON_DATASET
        or not isinstance(document.get("files"), Mapping)
    ):
        raise E1NativeBridgeError("E1 search manifest schema mismatch")
    observed = {
        str(row.get("path")): str(row.get("sha256"))
        for row in document["files"].values()
        if isinstance(row, Mapping)
    }
    if canonical_value(observed) != E1_SEARCH_FILE_SHA256:
        raise E1NativeBridgeError("E1 search manifest file identity drift")

    verified_files: dict[str, str] = {}
    if data_root is not None:
        root = Path(data_root).resolve()
        for relative, expected_sha256 in E1_SEARCH_FILE_SHA256.items():
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise E1NativeBridgeError("search file escapes data root") from error
            if not candidate.is_file() or _sha256(candidate.read_bytes()) != expected_sha256:
                raise E1NativeBridgeError(
                    f"search file identity mismatch: {relative}"
                )
            verified_files[relative] = expected_sha256
    return canonical_value(
        {
            "manifest_path": str(path),
            "manifest_sha256": _sha256(path.read_bytes()),
            "protocol_digest": E1_PROTOCOL_DIGEST,
            "verified_files": verified_files,
        }
    )


def _qualification_spec(
    candidate: E1NativeCandidate,
    *,
    hypothesis: str,
) -> OpenResearchSpecV1:
    profile_ref = "recclaw.e1.multvae-native-profile.v1"
    profile_digest = sha256_digest(
        {"profile_ref": profile_ref, "parent_id": E1_PARENT_ID}
    )
    context_ref = "recclaw.e1.mechanical-qualification-context.v1"
    context_digest = sha256_digest(
        {
            "context_ref": context_ref,
            "source_sha256": candidate.source_sha256,
        }
    )
    return OpenResearchSpecV1(
        hypothesis=hypothesis.strip() or "E1 candidate changes the active mechanism",
        mechanism_change="Execute the supplied candidate class instead of parent bytes.",
        competing_explanation="Any observed change may be implementation rather than mechanism.",
        matched_control_requirement="Compare with the exact frozen E1 parent.",
        implementation_requirements=(
            MODEL_CONFIG_MAPPING_REQUIREMENT,
            "RecBole GeneralRecommender interface",
            "user-wise multinomial reconstruction training",
            "finite loss and full-sort scores",
        ),
        expected_evidence=("one-epoch mechanical smoke result",),
        falsifier="Reject if the class fails the user-wise RecBole contract.",
        compatibility_requirements=(
            "general collaborative filtering",
            "offline top-n evaluation",
            "train-only fitting",
            "user-wise implicit-feedback input",
        ),
        protocol_ref=E1_PROTOCOL_REF,
        protocol_digest=E1_PROTOCOL_DIGEST,
        context_ref=context_ref,
        context_digest=context_digest,
        current_profile_ref=profile_ref,
        current_profile_digest=profile_digest,
        producer_role="frontier_architect",
        high_change_justification="Mechanical qualification does not rank the idea.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
        idea_mode=IdeaModeV1.FRONTIER_HYPOTHESIS,
        research_question="Does the candidate mechanism improve dev NDCG@10?",
        observed_failure_mode="NOT_OBSERVED",
        closest_parent=E1_PARENT_ID,
        minimal_testable_wedge="One candidate implementation under the fixed runner.",
        causal_chain=("candidate mechanism", "ranking behavior", "dev NDCG@10"),
        discriminative_predictions=("The candidate differs from exact parent bytes.",),
        mechanism_off_definition="Restore the exact frozen E1 parent package bytes.",
        resource_hypothesis="One user-wise VAE training remains inside the E1 cap.",
        realization_mode=RealizationModeV1.PARENT_PRESERVING,
        execution_contract={
            "capability_family": E1_CAPABILITY_FAMILY,
            "model": E1_MODEL,
            "base_model_config": E1_MODEL,
            "config": E1_TRAINING_CONFIG,
        },
    )


def _candidate_package(
    candidate: E1NativeCandidate,
    *,
    spec: OpenResearchSpecV1,
    runtime_identity_ref: str,
    runtime_identity_digest: str,
) -> CandidatePackageV1:
    receipt_identity = {
        "bridge_id": E1_BRIDGE_ID,
        "source_sha256": candidate.source_sha256,
    }
    return CandidatePackageV1(
        research_spec_ref=spec.spec_id,
        research_spec_digest=spec.digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        source_tree_digest=candidate.source_tree_digest,
        candidate_root_ref=candidate.candidate_root_ref,
        candidate_root_digest=candidate.candidate_root_digest,
        executable_entrypoint=E1_ENTRYPOINT,
        allowed_files=E1_ALLOWED_FILES,
        dependency_identity_ref="recclaw.e1.recbole-dependencies.v1",
        dependency_identity_digest=sha256_digest(
            {"recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"}
        ),
        runtime_identity_ref=runtime_identity_ref,
        runtime_identity_digest=runtime_identity_digest,
        implementation_receipt_ref=(
            "recclaw.e1.native-bridge-implementation:"
            + sha256_digest(receipt_identity)
        ),
        implementation_receipt_digest=sha256_digest(receipt_identity),
        origin_blind_projection_digest=sha256_digest(
            {"source_sha256": candidate.source_sha256}
        ),
    )


def qualify_candidate(
    candidate: E1NativeCandidate,
    *,
    repo_root: Path = _REPO_ROOT,
    qualification_root: Path,
    seed: int = 54201,
    hypothesis: str = "Mechanism-level E1 candidate",
    timeout_secs: int = 900,
    qualification_executor: Callable[..., MechanicalQualificationRun] | None = None,
) -> tuple[OpenResearchSpecV1, CandidatePackageV1, MechanicalQualificationRun]:
    """Run the existing five-stage qualifier with the E1 user-wise contract."""

    fixture = fresh_r1._qualification_fixture(
        Path(repo_root).resolve(),
        seed=seed,
        root=Path(qualification_root).resolve(),
        base_model_config=E1_MODEL,
    )
    spec = _qualification_spec(candidate, hypothesis=hypothesis)
    package = _candidate_package(
        candidate,
        spec=spec,
        runtime_identity_ref=fixture.runtime_identity_ref,
        runtime_identity_digest=fixture.runtime_identity_digest,
    )
    qualify = qualification_executor or MechanicalRecBoleAdapterV1().qualify_disposable
    qualification = qualify(
        package,
        research_spec=spec,
        candidate_root=candidate.candidate_root,
        fixture=fixture,
        timeout_seconds=timeout_secs,
    )
    return spec, package, qualification


def execution_recipe(
    candidate: E1NativeCandidate,
    *,
    spec: OpenResearchSpecV1,
    package: CandidatePackageV1,
    mechanism_id: str,
    epochs: int = 100,
) -> Mapping[str, Any]:
    """Build the exact dev-only recipe consumed by the trusted worker."""

    capability_ref = (
        "recclaw.e1.native-capability:"
        + sha256_digest(
            {
                "package_digest": package.digest,
                "mechanism_id": mechanism_id,
            }
        )
    )
    capability_digest = sha256_digest(
        {
            "capability_ref": capability_ref,
            "package_digest": package.digest,
            "entrypoint": candidate.entrypoint,
        }
    )
    return canonical_value(
        {
            "base_model_config": E1_MODEL,
            "candidate_package_digest": package.digest,
            "candidate_package_ref": package.package_id,
            "candidate_root_digest": candidate.candidate_root_digest,
            "candidate_root_ref": candidate.candidate_root_ref,
            "candidate_source_content_digest": candidate.file_sha256[E1_MODEL_RELATIVE_PATH],
            "candidate_source_tree_digest": candidate.source_tree_digest,
            "capability_digest": capability_digest,
            "capability_family": E1_CAPABILITY_FAMILY,
            "capability_ref": capability_ref,
            "config": {**E1_TRAINING_CONFIG, "epochs": epochs, "eval_step": 1},
            "dataset": COMMON_DATASET,
            "entrypoint": candidate.entrypoint,
            "entrypoint_source_sha256": candidate.file_sha256[E1_MODEL_RELATIVE_PATH],
            "evaluator": DEVELOPMENT_EVALUATOR,
            "evaluator_digest": sha256_digest(DEVELOPMENT_EVALUATOR),
            "execution_role": "CANDIDATE",
            "mechanism_id": mechanism_id,
            "model": E1_MODEL,
            "profile_digest": spec.current_profile_digest,
            "profile_ref": spec.current_profile_ref,
            "split": DEVELOPMENT_SPLIT,
        }
    )


def run_materialized_candidate(
    candidate: E1NativeCandidate,
    *,
    repo_root: Path = _REPO_ROOT,
    run_root: Path,
    run_id: str,
    seed: int = 54201,
    epochs: int = 100,
    timeout_secs: int = 3600,
    qualification_timeout_secs: int = 900,
    hypothesis: str = "Mechanism-level E1 candidate",
    mechanism_id: str | None = None,
    on_training_start: Callable[[Mapping[str, Any]], None] | None = None,
    gpu_id: int | None = None,
    cuda_visible_devices: str | None = None,
    search_manifest_path: Path = _DEFAULT_SEARCH_MANIFEST,
    verify_data_bytes: bool = True,
    qualification_executor: Callable[..., MechanicalQualificationRun] | None = None,
    process_launcher: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Qualify then train once; return only runner-authenticated dev metrics."""

    source_identity = candidate.source_sha256
    start_debited = False
    bridge_started = time.monotonic()
    try:
        manifest = verify_search_manifest(
            manifest_path=search_manifest_path,
            data_root=(fresh_r1.SEARCH_DATA_ROOT if verify_data_bytes else None),
        )
        spec, package, qualification = qualify_candidate(
            candidate,
            repo_root=repo_root,
            qualification_root=Path(run_root) / "qualification" / run_id,
            seed=seed,
            hypothesis=hypothesis,
            timeout_secs=qualification_timeout_secs,
            qualification_executor=qualification_executor,
        )
        if qualification.receipt.status is not QualificationStatusV1.PASS:
            return canonical_value(
                {
                    "status": "qualification_failed",
                    "ndcg_at_10": None,
                    "recall_at_10": None,
                    "wall_time_secs": 0.0,
                    "training_started": False,
                    "reused_completed_result": False,
                    "error_type": (
                        str(qualification.failure_detail.get("failure_class"))
                        if isinstance(qualification.failure_detail, Mapping)
                        else "QUALIFICATION"
                    ),
                    "error": qualification.failure_detail,
                    "artifacts": {
                        "qualification_receipt": qualification.receipt.canonical_dict(),
                        "search_manifest_sha256": manifest["manifest_sha256"],
                    },
                    "source_sha256": source_identity,
                }
            )
        effective_mechanism_id = mechanism_id or (
            "e1-mechanism-" + source_identity[:24]
        )
        recipe = execution_recipe(
            candidate,
            spec=spec,
            package=package,
            mechanism_id=effective_mechanism_id,
            epochs=epochs,
        )
        def debit_start(identity: Mapping[str, Any]) -> None:
            nonlocal start_debited
            if on_training_start is not None:
                on_training_start(canonical_value(dict(identity)))
            start_debited = True

        result = fresh_r1.run_development_training(
            repo_root=Path(repo_root).resolve(),
            side_root=Path(run_root).resolve(),
            run_id=run_id,
            seed=seed,
            candidate_root=candidate.candidate_root,
            entrypoint=candidate.entrypoint,
            source_sha256=candidate.file_sha256[E1_MODEL_RELATIVE_PATH],
            recbole_commit_identity=E1_RECBOLE_COMMIT,
            timeout_seconds=timeout_secs,
            epochs=epochs,
            execution_purpose="DEVELOPMENT_MAIN_OFFLINE_TOPN",
            execution_recipe=recipe,
            gpu_id=gpu_id,
            cuda_visible_devices=cuda_visible_devices,
            on_training_start=debit_start,
            **({"process_launcher": process_launcher} if process_launcher is not None else {}),
        )
        metrics = result.get("metrics")
        ndcg = metrics.get("ndcg@10") if isinstance(metrics, Mapping) else None
        recall = metrics.get("recall@10") if isinstance(metrics, Mapping) else None
        trusted_success = (
            result.get("exit_status") == "SUCCESS"
            and result.get("metric_source") == "BEST_VALID_RESULT"
            and result.get("online_partition_role") == "DEVELOPMENT_VALIDATION"
            and isinstance(ndcg, (int, float))
            and not isinstance(ndcg, bool)
            and math.isfinite(float(ndcg))
            and isinstance(recall, (int, float))
            and not isinstance(recall, bool)
            and math.isfinite(float(recall))
        )
        wall_time = result.get("reserved_gpu_worker_seconds", 0.0)
        if not isinstance(wall_time, (int, float)) or isinstance(wall_time, bool):
            wall_time = 0.0
        return canonical_value(
            {
                "status": "ok" if trusted_success else "training_failed",
                "ndcg_at_10": float(ndcg) if trusted_success else None,
                "recall_at_10": float(recall) if trusted_success else None,
                "wall_time_secs": float(wall_time) if start_debited else 0.0,
                "training_started": start_debited,
                "reused_completed_result": trusted_success and not start_debited,
                "error_type": None if trusted_success else result.get("worker_error_type"),
                "error": None if trusted_success else result.get("worker_error_message"),
                "artifacts": {
                    "candidate_root_digest": candidate.candidate_root_digest,
                    "checkpoint_footprint": result.get("checkpoint_footprint"),
                    "experiment_binding_digest": result.get(
                        "experiment_binding_digest"
                    ),
                    "qualification_receipt_digest": qualification.receipt.digest,
                    "result_sha256": result.get("result_sha256"),
                    "search_manifest_sha256": manifest["manifest_sha256"],
                },
                "source_sha256": source_identity,
            }
        )
    except Exception as error:  # The bridge returns one stable controller ABI.
        error_type = type(error).__name__
        return canonical_value(
            {
                "status": (
                    "budget_exhausted"
                    if "budget" in error_type.lower()
                    else "bridge_error"
                ),
                "ndcg_at_10": None,
                "recall_at_10": None,
                "wall_time_secs": (
                    time.monotonic() - bridge_started if start_debited else 0.0
                ),
                "training_started": start_debited,
                "reused_completed_result": False,
                "error_type": error_type,
                "error": str(error),
                "artifacts": {
                    "candidate_root_digest": candidate.candidate_root_digest,
                },
                "source_sha256": source_identity,
            }
        )


def run_candidate_path(
    *,
    candidate_path: str | Path,
    timeout_secs: int,
    on_training_start: Callable[[Mapping[str, Any]], None] | None,
    run_root: Path,
    repo_root: Path = _REPO_ROOT,
    seed: int = 54201,
    epochs: int = 100,
    qualification_timeout_secs: int = 900,
    run_id_prefix: str = "e1-candidate",
    hypothesis: str = "Mechanism-level E1 candidate",
    gpu_id: int | None = None,
    cuda_visible_devices: str | None = None,
    search_manifest_path: Path = _DEFAULT_SEARCH_MANIFEST,
    verify_data_bytes: bool = True,
    qualification_executor: Callable[..., MechanicalQualificationRun] | None = None,
    process_launcher: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Controller-neutral ``best.py`` entry into the shared native runner.

    ``run_root`` is trusted caller state, intentionally separate from the
    controller-owned candidate workspace.  Exact source bytes determine the
    materialization and run identity, so an interrupted completed run can be
    replayed without another physical training start.
    """

    path = Path(candidate_path).resolve()
    if not path.is_file():
        raise E1NativeBridgeError("candidate_path does not name a file")
    source_payload = path.read_bytes()
    _model_source(source_payload)
    source_sha256 = _sha256(source_payload)
    if (
        not isinstance(run_id_prefix, str)
        or not run_id_prefix
        or run_id_prefix != run_id_prefix.strip()
        or "/" in run_id_prefix
        or "\\" in run_id_prefix
        or run_id_prefix in {".", ".."}
    ):
        raise E1NativeBridgeError("run_id_prefix must be one normalized path atom")
    run_id = f"{run_id_prefix}-s{seed}-e{epochs}-{source_sha256[:24]}"
    trusted_root = Path(run_root).resolve()
    materialized_root = trusted_root / "materialized" / run_id
    if materialized_root.exists():
        candidate = _load_materialized_candidate(
            candidate_root=materialized_root,
            expected_source_payload=source_payload,
        )
    else:
        candidate = materialize_candidate_source(
            source_path=path,
            candidate_root=materialized_root,
            expected_source_sha256=source_sha256,
        )
    return run_materialized_candidate(
        candidate,
        repo_root=repo_root,
        run_root=trusted_root,
        run_id=run_id,
        seed=seed,
        epochs=epochs,
        timeout_secs=timeout_secs,
        qualification_timeout_secs=qualification_timeout_secs,
        hypothesis=hypothesis,
        mechanism_id="e1-mechanism-" + source_sha256[:24],
        on_training_start=on_training_start,
        gpu_id=gpu_id,
        cuda_visible_devices=cuda_visible_devices,
        search_manifest_path=search_manifest_path,
        verify_data_bytes=verify_data_bytes,
        qualification_executor=qualification_executor,
        process_launcher=process_launcher,
    )


def make_native_runner(
    *,
    run_root: Path,
    repo_root: Path = _REPO_ROOT,
    seed: int = 54201,
    epochs: int = 100,
    qualification_timeout_secs: int = 900,
    run_id_prefix: str = "e1-candidate",
    gpu_id: int | None = None,
    cuda_visible_devices: str | None = None,
    search_manifest_path: Path = _DEFAULT_SEARCH_MANIFEST,
    verify_data_bytes: bool = True,
    qualification_executor: Callable[..., MechanicalQualificationRun] | None = None,
    process_launcher: Callable[..., Any] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """Bind trusted machine inputs and expose the exact two-controller ABI."""

    def native_runner(
        *,
        candidate_path: str,
        timeout_secs: int,
        on_training_start: Callable[[Mapping[str, Any]], None] | None,
    ) -> Mapping[str, Any]:
        return run_candidate_path(
            candidate_path=candidate_path,
            timeout_secs=timeout_secs,
            on_training_start=on_training_start,
            run_root=run_root,
            repo_root=repo_root,
            seed=seed,
            epochs=epochs,
            qualification_timeout_secs=qualification_timeout_secs,
            run_id_prefix=run_id_prefix,
            gpu_id=gpu_id,
            cuda_visible_devices=cuda_visible_devices,
            search_manifest_path=search_manifest_path,
            verify_data_bytes=verify_data_bytes,
            qualification_executor=qualification_executor,
            process_launcher=process_launcher,
        )

    return native_runner


__all__ = [
    "E1_ALLOWED_FILES",
    "E1_BRIDGE_ID",
    "E1_ENTRYPOINT",
    "E1_MODEL",
    "E1_MODEL_RELATIVE_PATH",
    "E1NativeBridgeError",
    "E1NativeCandidate",
    "E1_PARENT_ID",
    "E1_PROTOCOL",
    "E1_PROTOCOL_DIGEST",
    "E1_PROTOCOL_REF",
    "E1_RECBOLE_COMMIT",
    "E1_SEARCH_FILE_SHA256",
    "E1_TRAINING_CONFIG",
    "execution_recipe",
    "make_native_runner",
    "materialize_candidate_source",
    "parent_package_identity",
    "qualify_candidate",
    "run_candidate_path",
    "run_materialized_candidate",
    "verify_search_manifest",
]
