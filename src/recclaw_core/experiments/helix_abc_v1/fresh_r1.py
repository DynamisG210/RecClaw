"""Formal fresh R1 orchestration over the accepted v11 OpenSpec contract.

This module composes existing Provider, OpenSpec, Resolver, Innovation Spine,
Qualifier, admission, RecBole worker, and Episode contracts.  It does not add a
candidate catalog, fallback path, mutable service, or post-outcome repair path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import socket
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema

from .canary_broker import CanaryBrokerError, CanaryBrokerCallV1
from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from .capability_admission import admit_qualified_capability
from .experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    ExperimentBindingError,
    ExperimentBindingV1,
    render_campaign_worker_command,
    validate_execution_recipe,
)
from .innovation_recbole_adapter import (
    MechanicalQualificationRun,
    MechanicalRecBoleAdapterV1,
    RecBoleQualificationFixture,
    snapshot_candidate_tree,
)
from .innovation_spine import (
    InnovationSpineError,
    MaterializedCandidate,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from .lab_api_broker import (
    LabApiCanaryBrokerV1,
    load_lab_api_credential_pairs,
    load_lab_api_credentials,
)
from .open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
    project_open_producer_draft,
    resolve_capability,
)
from .training_filesystem import (
    build_training_filesystem_capability,
    materialize_training_filesystem_capability,
)
from .training_runtime_release import (
    CAMPAIGN_TRAINING_RUNNER_ABI,
    campaign_training_runtime_release,
)
from .v4_response_contract import (
    V4LocalUniquenessError,
    validate_v4_response_contract,
)
from .vnext_contracts import (
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    EpisodeEvidenceClassV1,
    QualificationStatusV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)


ACCEPTED_COMMIT = "e43bb78acbfbe1e616cc320cf9ff6b55b2144286"
ACCEPTED_PARENT = "dd75f8f50d1c9e26593bf60171a9b3019ad6420b"
ACCEPTED_TREE = "b058b17cd1c35e68b2402444b66cfd01deb9ff07"
READY_SHA256 = "aede1d9fbbded5b0145e4ff48ff6d7d19d2bdbb40959dfe73c61fa5bb8cf8de9"
MANIFEST_SHA256 = "b021e49d24d3b1cbe8d9cec52fc902351d46f975b34d31bf18d32ecfe2941a9a"
MODEL = "gpt-5.4"
PROPOSAL_TOKEN_CEILING = 6000
IMPLEMENTATION_TOKEN_CEILING = 20_000
EXPERIMENT_EPOCHS = 100
BACKOFF_MS = (1000, 3000)
MAX_PHYSICAL_ATTEMPTS = 3
MAX_WORKER_CEILING_SECONDS = 3600
GPU_RESERVATION_EVIDENCE_SCHEMA = "recclaw.gpu-reservation-evidence.v1"
GPU_RESERVATION_STATUS_MEASURED = (
    "MEASURED_RESERVATION_SCOPED_PROCESS_INTERVAL_NOT_GPU_ACTIVE"
)
GPU_RESERVATION_STATUS_UNMEASURED = "UNMEASURED_NO_EXCLUSIVE_RESERVATION"
GPU_RESERVATION_STATUS_CONTRADICTORY = (
    "UNMEASURED_CONTRADICTORY_DEVICE_EVIDENCE"
)
GPU_WORKER_SECONDS_SEMANTICS = (
    "RESERVATION_SCOPED_PARENT_PROCESS_INTERVAL; NOT_GPU_ACTIVE_UTILIZATION"
)
DIRECT_GPU_SELECTION_MODE = "DIRECT_RECOBOLE_GPU_ID"
_GPU_RESERVATION_SCOPES = frozenset(
    {"CANDIDATE_WORKER", "ONE_CANDIDATE_ONE_WORKER"}
)
DEFAULT_NATIVE_STOPPING_PATIENCE = 10
HEALTH_ACTION_CONTINUE = "CONTINUE"
HEALTH_ACTION_NATIVE_SEAL = "NATIVE_SEAL"
HEALTH_ACTION_HEALTH_ABORT = "HEALTH_ABORT"
HEALTH_ACTION_HARD_CEILING = "HARD_CEILING"
HEALTH_STATE_STARTING = "STARTING"
HEALTH_STATE_INITIALIZED = "INITIALIZED"
HEALTH_STATE_TRAIN_PROGRESS = "TRAIN_PROGRESS"
HEALTH_STATE_VALIDATION_PROGRESS = "VALIDATION_PROGRESS"
HEALTH_STATE_NATIVE_SEAL = "NATIVE_SEAL"
HEALTH_STATE_HEALTH_ABORT = "HEALTH_ABORT"
HEALTH_STATE_HARD_CEILING = "HARD_CEILING"
CONTEXT_REF = "fresh-r1-r2-prefreeze-context-v1"
CONTEXT_DIGEST = "4965758687e260c490e4f103910cf683d6d3b695c631d67ee27e0385c5704bce"
R1_ROOT = Path(
    "/root/projects/RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
)
SEALED_R1_RECEIPT = Path(
    "/root/projects/RecClaw_fresh_r1/docs/research_line/vnext/"
    "R1_FRESH_CANONICAL_RECEIPT.json"
)
SEALED_R1_RECEIPT_SHA256 = (
    "e22fc697df65b501e783de6b95cdafc537581f49a76dfd6a8f818052cc0879c2"
)
SEALED_R1_EXTERNAL_RECEIPT = Path(
    "/root/projects/RecClaw_r1_r2_runs/fresh_r1_open_spec_v1/"
    "R1_CANONICAL_RECEIPT.json"
)
SEALED_R1_EXTERNAL_RECEIPT_SHA256 = (
    "446d53611bdcc21d685b080bcf1be158d124a75973d93ed390511817c64bbdda"
)
SEALED_CORRECTED_R1_RECEIPT = Path(
    "/root/projects/RecClaw_fresh_r1/docs/research_line/vnext/"
    "R1_FRESH_CORRECTED_CANONICAL_RECEIPT.json"
)
SEALED_CORRECTED_R1_RECEIPT_SHA256 = (
    "c3c7bee005d06539474020af72dd79c8ca9d818818c7cea09de80278c3e7f4df"
)
SEALED_CORRECTED_R1_EXTERNAL_RECEIPT = Path(
    "/root/projects/RecClaw_r1_r2_runs/fresh_r1_open_spec_corrected_v1/"
    "R1_CANONICAL_RECEIPT.json"
)
SEALED_CORRECTED_R1_EXTERNAL_RECEIPT_SHA256 = (
    "a45ad3a13147acdd5b14f9988b28d4d789dce6b2e2d5b27d4803756ed9607431"
)
PROJECTS_ROOT = Path(os.environ.get("RECCLAW_PROJECTS_ROOT", "/root/projects"))
SEARCH_DATA_ROOT = Path(
    os.environ.get(
        "RECCLAW_SEARCH_DATA_ROOT",
        str(PROJECTS_ROOT / "RecClaw_campaign_dataset_v1/search"),
    )
)
SEARCH_DATASET_ROOT = SEARCH_DATA_ROOT / "ml-1m"
RECBole_ROOT = Path(
    os.environ.get("RECCLAW_RECBOLE_ROOT", str(PROJECTS_ROOT / "RecBole"))
)
PYTHON_EXECUTABLE = Path(
    os.environ.get(
        "RECCLAW_PYTHON_EXECUTABLE",
        "/root/miniconda3/envs/recbole/bin/python3.10",
    )
)
API_CONFIG = Path(
    os.environ.get(
        "RECCLAW_API_CONFIG",
        str(PROJECTS_ROOT / "RecClaw_v2_0_Final_Reference/llm_api.md"),
    )
)
EXPECTED_SEARCH_FILES = {
    "ml-1m.train.inter": "c84b1a4f6c6d974f32f126b173f11f7af8e12e1a143a50ac5a53e9945903491a",
    "ml-1m.dev.inter": "631911b8e59d312e110ba7205151bcda3d52378ecbf510d48d8b9cc162956cc9",
}
AVAILABLE_DEPENDENCIES = (
    "numpy",
    "python-stdlib",
    "pytorch",
    "recbole-runtime",
    "scipy",
    "torch",
)
BUDGET_LIMITS = {
    "implementation_token_ceiling": 20_000,
    "qualification_gpu_minutes": 10,
    "qualification_wall_minutes": 30,
}
PROTOCOL_REQUIREMENTS = (
    "frozen dataset and split",
    "full-sort NDCG@10",
    "general collaborative filtering",
    "offline top-n evaluation",
    "pairwise input",
    "train-only fitting",
)
ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "Compose a genuinely new multi-part scoring or learning mechanism. "
        "Its compatibility_requirements must be a subset of the exact frozen "
        "protocol requirements listed below, and dependencies a subset of the "
        "exact available dependencies."
    ),
    "lineage_refiner": (
        "Start from a mechanistic weakness, then propose a new executable "
        "descendant whose behavior is not a catalog composition. Keep protocol "
        "and dependency declarations within the exact lists below."
    ),
    "falsification_designer": (
        "Design a mechanism around a decisive matched-control falsifier, with "
        "a real trainable behavior change. Keep protocol and dependency "
        "declarations within the exact lists below."
    ),
    "frontier_architect": (
        "Propose a new structural frontier for pairwise recommendation, not a "
        "wrapper or hyperparameter change. Keep protocol and dependency "
        "declarations within the exact lists below."
    ),
}

CORRECTED_RUN_IDENTITY = "fresh-r1-training-filesystem-fix-v3"


class FreshR1Error(RuntimeError):
    """A run-level identity, protocol, or orchestration failure."""


@dataclass(frozen=True, slots=True)
class ProviderAttemptResult:
    call: CanaryBrokerCallV1 | None
    attempts: tuple[Mapping[str, Any], ...]
    failure: Mapping[str, Any] | None


def _resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise FreshR1Error(f"JSON artifact is not an object: {path}")
    return value


def derive_fresh_r1_proposal_schema(
    base_schema: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the two exact-token restrictions authorized for corrected R1."""

    if (
        delta.get("schema")
        != "recclaw.research-line.fresh-r1-proposal-schema-delta.v1"
        or tuple(delta.get("compatibility_requirement_tokens", ()))
        != PROTOCOL_REQUIREMENTS
        or tuple(delta.get("required_dependency_tokens", ()))
        != AVAILABLE_DEPENDENCIES
    ):
        raise FreshR1Error("corrected R1 proposal schema delta changed")
    derived = canonical_value(base_schema)
    proposal = derived["properties"]["proposals"]["items"]
    compatibility_items = proposal["properties"]["compatibility_requirements"][
        "items"
    ]
    dependency_items = proposal["properties"]["resolution_facts"]["properties"][
        "required_dependencies"
    ]["items"]
    compatibility_items["enum"] = list(PROTOCOL_REQUIREMENTS)
    dependency_items["enum"] = list(AVAILABLE_DEPENDENCIES)
    jsonschema.validators.validator_for(derived).check_schema(derived)
    return derived


def call_contract_for_side(
    side: str,
    *,
    service: str,
    prompt_digest: str,
    response_schema_digest: str,
) -> dict[str, Any]:
    """Return the side-independent physical call contract for one R1 service."""

    if side not in {"side_a", "side_b"}:
        raise FreshR1Error(f"unknown R1 side: {side}")
    ceilings = {
        "proposal": PROPOSAL_TOKEN_CEILING,
        "implementation": IMPLEMENTATION_TOKEN_CEILING,
    }
    if service not in ceilings:
        raise FreshR1Error(f"unknown R1 Provider service: {service}")
    return canonical_value(
        {
            "granularity": "ONE_PROPOSAL_PER_LOGICAL_CALL",
            "model": MODEL,
            "prompt_digest": prompt_digest,
            "response_schema_digest": response_schema_digest,
            "temperature": 0,
            "token_ceiling": ceilings[service],
            "tools": [],
        }
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def recbole_source_identity(recbole_root: Path = RECBole_ROOT) -> dict[str, Any]:
    """Content-bind the RecBole Python/config tree actually used by a run."""

    root = Path(recbole_root).resolve()
    rows = tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": bytes_sha256(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml"}
    )
    if not rows:
        raise FreshR1Error("RecBole source tree contains no Python/config files")
    return canonical_value(
        {
            "file_count": len(rows),
            "source_root": str(root),
            "source_tree_digest": sha256_digest(rows),
        }
    )


def _credential_identity(config_path: Path) -> dict[str, str]:
    base_url, api_key = load_lab_api_credentials(config_path)
    config_digest = bytes_sha256(config_path.read_bytes())
    endpoint_digest = sha256_digest({"base_url": base_url.rstrip("/")})
    identity_digest = sha256_digest(
        {
            "scheme": "RECClaw credential config identity v1",
            "config_bytes_sha256": config_digest,
            "api_key_sha256": hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            "endpoint_digest": endpoint_digest,
        }
    )
    del api_key
    return {
        "credential_config_digest": config_digest,
        "credential_identity_digest": identity_digest,
        "endpoint_digest": endpoint_digest,
    }


def verify_formal_identity(repo_root: Path) -> dict[str, Any]:
    docs = repo_root / "docs/research_line/vnext"
    ready_path = docs / "R1_PREFREEZE_READY_RECEIPT.json"
    manifest_path = docs / "R1_R2_PREFREEZE_MANIFEST_V11.json"
    ready = _read_json(ready_path)
    manifest = _read_json(manifest_path)
    observed = {
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "parent": _git(repo_root, "rev-parse", "HEAD^"),
        "head_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "ready_sha256": bytes_sha256(ready_path.read_bytes()),
        "manifest_sha256": bytes_sha256(manifest_path.read_bytes()),
        "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
        "sealed_corrected_r1_external_receipt_sha256": bytes_sha256(
            SEALED_CORRECTED_R1_EXTERNAL_RECEIPT.read_bytes()
        ),
        "sealed_corrected_r1_receipt_sha256": bytes_sha256(
            SEALED_CORRECTED_R1_RECEIPT.read_bytes()
        ),
        "sealed_r1_external_receipt_sha256": bytes_sha256(
            SEALED_R1_EXTERNAL_RECEIPT.read_bytes()
        ),
        "sealed_r1_receipt_sha256": bytes_sha256(SEALED_R1_RECEIPT.read_bytes()),
        **_credential_identity(API_CONFIG),
    }
    expected = {
        "head": ACCEPTED_COMMIT,
        "parent": ACCEPTED_PARENT,
        "head_tree": ACCEPTED_TREE,
        "branch": "feat/research-line-fresh-r1",
        "ready_sha256": READY_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "python_sha256": "d99cded726bcf8b1576305ef425915fc7009c40325ecc2659553ab1c94997938",
        "sealed_corrected_r1_external_receipt_sha256": (
            SEALED_CORRECTED_R1_EXTERNAL_RECEIPT_SHA256
        ),
        "sealed_corrected_r1_receipt_sha256": SEALED_CORRECTED_R1_RECEIPT_SHA256,
        "sealed_r1_external_receipt_sha256": SEALED_R1_EXTERNAL_RECEIPT_SHA256,
        "sealed_r1_receipt_sha256": SEALED_R1_RECEIPT_SHA256,
        "credential_config_digest": manifest["exact_provider_contract"][
            "credential_config_digest"
        ],
        "credential_identity_digest": manifest["exact_provider_contract"][
            "credential_identity_digest"
        ],
        "endpoint_digest": manifest["exact_provider_contract"]["endpoint_digest"],
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise FreshR1Error("formal R1 identity mismatch: " + json.dumps(mismatches, sort_keys=True))
    if (
        ready.get("status") != "R1_PREFREEZE_READY"
        or ready.get("r1_worker_launch_authorized") is not True
        or ready.get("model") != MODEL
        or ready.get("manifest_digest") != MANIFEST_SHA256
    ):
        raise FreshR1Error("READY receipt does not authorize exact frozen R1")
    if _git(RECBole_ROOT, "rev-parse", "HEAD") != "7b02be5ec80a88310f2d04a27a82adfcbb5dc211":
        raise FreshR1Error("RecBole commit identity mismatch")
    if _git(RECBole_ROOT, "rev-parse", "HEAD^{tree}") != "ca6386c4121ce2aae478ced7e136894ac1d7c218":
        raise FreshR1Error("RecBole tree identity mismatch")
    for name, digest in EXPECTED_SEARCH_FILES.items():
        if bytes_sha256((SEARCH_DATASET_ROOT / name).read_bytes()) != digest:
            raise FreshR1Error(f"search partition identity mismatch: {name}")
    if R1_ROOT.exists():
        raise FreshR1Error(f"fresh R1 root already exists: {R1_ROOT}")
    return canonical_value(
        {
            **observed,
            "ready": {
                "attempt_id": ready["attempt_id"],
                "model": ready["model"],
                "status": ready["status"],
            },
            "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
            "recbole_tree": "ca6386c4121ce2aae478ced7e136894ac1d7c218",
            "search_partition_files": EXPECTED_SEARCH_FILES,
        }
    )


def render_proposal_prompt(
    template: str,
    *,
    side_identity: str,
    logical_slot_id: str,
    proposal_seed: int,
    producer_role: str,
) -> str:
    instruction = ROLE_INSTRUCTIONS[producer_role] + (
        " Each compatibility_requirements array item MUST be copied verbatim "
        "as one token from this list; do not emit a sentence, paraphrase, or "
        "combined clause: "
        + ", ".join(PROTOCOL_REQUIREMENTS)
        + ". Each resolution_facts.required_dependencies array item MUST be "
        "copied verbatim as one token from this list; do not emit a sentence, "
        "paraphrase, or combined clause: "
        + ", ".join(AVAILABLE_DEPENDENCIES)
        + ". Required budgets may not exceed "
        + json.dumps(BUDGET_LIMITS, sort_keys=True)
        + "."
    )
    replacements = {
        "{{SIDE_IDENTITY}}": side_identity,
        "{{LOGICAL_SLOT_ID}}": logical_slot_id,
        "{{PROPOSAL_SEED}}": str(proposal_seed),
        "{{PRODUCER_ROLE}}": producer_role,
        "{{PRODUCER_ROLE_INSTRUCTION}}": instruction,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "{{" in rendered or "}}" in rendered:
        raise FreshR1Error("proposal prompt has an unresolved placeholder")
    return rendered


def render_implementation_prompt(template: str, request: Mapping[str, Any]) -> str:
    placeholder = "{{SHARED_IMPLEMENTER_REQUEST_JSON}}"
    rendered = template.replace(
        placeholder,
        canonical_json_bytes(request).decode("utf-8"),
    )
    if placeholder in rendered:
        raise FreshR1Error("implementation prompt has an unresolved placeholder")
    return rendered


def _attempt_failure(private_root: Path, error: CanaryBrokerError) -> dict[str, Any]:
    outcome = error.outcome
    request_digest = (
        outcome.request_envelope_digest if outcome is not None else None
    )
    http_status = (
        error.receipt.exit_code_or_NONE if error.receipt is not None else None
    )
    exception_type = None
    receipt: dict[str, Any] = {}
    db_path = private_root / "broker.sqlite3"
    if db_path.is_file():
        connection = sqlite3.connect(db_path)
        try:
            if outcome is not None:
                row = connection.execute(
                    "SELECT error_detail_json, receipt_json FROM calls "
                    "WHERE logical_call_id=?",
                    (outcome.logical_call_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT error_detail_json, receipt_json FROM calls "
                    "ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
        finally:
            connection.close()
        if row is not None:
            detail = json.loads(str(row[0])) if row[0] else {}
            receipt = json.loads(str(row[1])) if row[1] else {}
            exception_type = detail.get("exception_type")
            if http_status is None:
                http_status = receipt.get("http_status")
            if request_digest is None:
                request_digest = receipt.get("request_digest")
    return canonical_value(
        {
            "error_type": type(error).__name__,
            "exception_type": exception_type,
            "failure_class": (
                error.outcome.failure_class if error.outcome is not None else "LOCAL_BROKER_FAILURE"
            ),
            "http_status": http_status,
            "message": str(error),
            "mechanism_negative_evidence": False,
            "physical_call_count": error.physical_call_count,
            "request_digest": request_digest,
            "receipt_digest": (
                error.receipt.receipt_digest if error.receipt is not None else None
            ),
        }
    )


def retry_eligible(failure: Mapping[str, Any]) -> bool:
    status = failure.get("http_status")
    if isinstance(status, int) and (status in {401, 403, 408, 429} or status >= 500):
        return True
    if failure.get("failure_class") == "TIMEOUT":
        return True
    return failure.get("exception_type") in {
        "ConnectionResetError",
        "RemoteDisconnected",
    }


def _credential_schedule(
    *,
    config_path: Path,
    credential_index: int | None,
    credential_schedule: Sequence[int] | None,
) -> tuple[int, ...]:
    pairs = load_lab_api_credential_pairs(config_path)
    if credential_schedule is None:
        schedule = (
            (credential_index,)
            if credential_index is not None
            else tuple(range(len(pairs)))
        )
    else:
        schedule = tuple(credential_schedule)
        if credential_index is not None and (
            not schedule or schedule[0] != credential_index
        ):
            raise FreshR1Error(
                "credential_index does not match credential_schedule"
            )
    if not schedule:
        raise FreshR1Error("credential schedule must not be empty")
    for index in schedule:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise FreshR1Error("credential schedule contains an invalid index")
        if index >= len(pairs):
            raise FreshR1Error(
                "credential schedule index is outside the configured pairs"
            )
    return schedule


def _expected_release_digest(
    expected: str | Sequence[str] | None,
    *,
    credential_index: int,
    schedule_position: int,
) -> str | None:
    if expected is None:
        return None
    if isinstance(expected, str):
        # The legacy scalar identifies the first physical credential. A
        # per-index sequence pins fallback releases too.
        return expected if schedule_position == 0 else None
    expected_by_index = tuple(expected)
    if credential_index >= len(expected_by_index):
        return None
    value = expected_by_index[credential_index]
    if not isinstance(value, str):
        raise FreshR1Error("expected transport release digest is invalid")
    return value


def bounded_provider_call(
    *,
    call_root: Path,
    schema_path: Path,
    logical_call_id: str,
    session_id: str,
    prompt: str,
    token_ceiling: int,
    output_token_ceiling: int | None = None,
    expected_transport_release_digest: str | Sequence[str] | None = None,
    credential_config_path: Path | None = None,
    credential_index: int | None = None,
    credential_schedule: Sequence[int] | None = None,
    maximum_physical_attempts: int = MAX_PHYSICAL_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> ProviderAttemptResult:
    if not isinstance(maximum_physical_attempts, int) or not (
        1 <= maximum_physical_attempts <= MAX_PHYSICAL_ATTEMPTS
    ):
        raise FreshR1Error("maximum_physical_attempts is outside the bounded policy")
    config_path = Path(credential_config_path or API_CONFIG).resolve()
    schedule = _credential_schedule(
        config_path=config_path,
        credential_index=credential_index,
        credential_schedule=credential_schedule,
    )
    attempts: list[dict[str, Any]] = []
    request_digests: set[str] = set()
    for ordinal in range(1, maximum_physical_attempts + 1):
        if ordinal > 1:
            sleep(BACKOFF_MS[ordinal - 2] / 1000)
        schedule_position = min(ordinal - 1, len(schedule) - 1)
        selected_credential_index = schedule[schedule_position]
        private_root = call_root / f"physical_attempt_{ordinal:02d}"
        broker: LabApiCanaryBrokerV1 | None = None
        started = time.monotonic_ns()
        try:
            broker = LabApiCanaryBrokerV1(
                private_root,
                schema_path=schema_path,
                config_path=config_path,
                model=MODEL,
                max_total_tokens_per_call=token_ceiling,
                credential_index=selected_credential_index,
                timeout_ms=900_000,
                release_manifest_path=None,
            )
            expected_release_digest = _expected_release_digest(
                expected_transport_release_digest,
                credential_index=selected_credential_index,
                schedule_position=schedule_position,
            )
            if (
                expected_release_digest is not None
                and broker.release.release_digest != expected_release_digest
            ):
                raise FreshR1Error("proposal transport release identity mismatch")
            call = broker.call_with_session(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=session_id,
                prompt=prompt,
                expected_proposal_count=1,
                max_total_tokens=token_ceiling,
                max_output_tokens=output_token_ceiling,
            )
            request_digests.add(call.request_digest)
            if len(request_digests) != 1:
                raise FreshR1Error("bounded retry changed the request payload digest")
            attempts.append(
                {
                    "billed_tokens": call.total_tokens,
                    "credential_config_digest": broker.credential_config_digest,
                    "credential_identity_digest": broker.credential_identity_digest,
                    "credential_index": broker.credential_index,
                    "endpoint_digest": broker.endpoint_digest,
                    "input_tokens": call.input_tokens,
                    "latency_ms": call.latency_ms,
                    "ordinal": ordinal,
                    "output_tokens": call.output_tokens,
                    "request_digest": call.request_digest,
                    "returned_model": call.returned_model,
                    "status": "SUCCESS",
                }
            )
            return ProviderAttemptResult(
                call=call,
                attempts=tuple(canonical_value(attempts)),
                failure=None,
            )
        except FreshR1Error:
            raise
        except CanaryBrokerError as error:
            failure = _attempt_failure(private_root, error)
            if failure.get("request_digest"):
                request_digests.add(str(failure["request_digest"]))
            if len(request_digests) != 1:
                raise FreshR1Error("bounded retry changed the request payload digest")
            attempts.append(
                {
                    **failure,
                    "credential_config_digest": (
                        broker.credential_config_digest
                        if broker is not None
                        else None
                    ),
                    "credential_identity_digest": (
                        broker.credential_identity_digest
                        if broker is not None
                        else None
                    ),
                    "credential_index": selected_credential_index,
                    "endpoint_digest": (
                        broker.endpoint_digest if broker is not None else None
                    ),
                    "latency_ms": max(1, (time.monotonic_ns() - started) // 1_000_000),
                    "ordinal": ordinal,
                    "status": "FAILED",
                }
            )
            if not retry_eligible(failure) or ordinal == maximum_physical_attempts:
                return ProviderAttemptResult(
                    call=None,
                    attempts=tuple(canonical_value(attempts)),
                    failure=failure,
                )
        finally:
            if broker is not None:
                broker.close()
    raise AssertionError("bounded Provider loop did not terminate")


def evaluate_gate(side_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    side_results: dict[str, Any] = {}
    structural_total = 0
    for side, records in sorted(side_records.items()):
        fresh_specs = [row for row in records if row.get("spec_digest")]
        roles = sorted({str(row["producer_role"]) for row in fresh_specs})
        qualified = [row for row in records if row.get("qualification_status") == "PASS"]
        structural = [row for row in qualified if row.get("real_mechanism_change") is True]
        structural_total += len(structural)
        side_results[side] = {
            "fresh_spec_count": len(fresh_specs),
            "producer_roles": roles,
            "producer_role_count": len(roles),
            "qualified_capability_count": len(qualified),
            "real_mechanism_change_count": len(structural),
            "pass": len(fresh_specs) >= 4 and len(roles) >= 2 and len(qualified) >= 2,
        }
    passed = (
        set(side_results) == {"side_a", "side_b"}
        and all(row["pass"] for row in side_results.values())
        and structural_total >= 1
    )
    return {
        "accepted_change_kinds": ["STRUCTURAL", "INTERACTION", "PROPAGATION"],
        "minimum_fresh_specs_per_side": 4,
        "minimum_producer_roles_per_side": 2,
        "minimum_qualified_capabilities_per_side": 2,
        "minimum_real_mechanism_changes_overall": 1,
        "pass": passed,
        "side_results": side_results,
        "structural_change_count_overall": structural_total,
    }


def _mechanism_kind(dimensions: Sequence[str]) -> CapabilityKindV1:
    values = set(dimensions)
    if "PROPAGATION_MECHANISM" in values:
        return CapabilityKindV1.PROPAGATION_MECHANISM
    if "INTERACTION_STRUCTURE" in values:
        return CapabilityKindV1.INTERACTION_HEAD
    if "MODEL_STRUCTURE" in values or "CORE_REPRESENTATION" in values:
        return CapabilityKindV1.COMPLETE_MODEL
    return CapabilityKindV1.COMPOSITE_MODULE


def _shared_behavioral_unit_check(
    evidence: dict[str, Any],
    *,
    require_extra_parameters: bool = True,
    base_model_config: str = "BPR",
) -> Callable[[Any, Any, Any], None]:
    def check(model: Any, config: Any, dataset: Any) -> None:
        import torch

        from recbole.data.interaction import Interaction
        from recbole.model.general_recommender.bpr import BPR
        from recbole.model.general_recommender.lightgcn import LightGCN
        from recbole.utils import InputType, ModelType

        reference_classes = {"BPR": BPR, "LightGCN": LightGCN}
        if base_model_config not in reference_classes:
            raise AssertionError(
                "family-neutral qualifier requires a known frozen base configuration"
            )
        reference_class = reference_classes[base_model_config]

        if config["MODEL_TYPE"] is not ModelType.GENERAL:
            raise AssertionError("candidate is not a general recommender")
        if model.input_type is not InputType.PAIRWISE:
            raise AssertionError("candidate is not pairwise")
        if base_model_config == "LightGCN" and isinstance(model, BPR):
            raise AssertionError(
                "LightGCN execution contract cannot be implemented as a BPR subclass"
            )
        overridden = tuple(
            name
            for name in ("calculate_loss", "predict", "full_sort_predict")
            if name in model.__class__.__dict__
        )
        if len(overridden) != 3:
            raise AssertionError("candidate must implement all three behavioral methods")
        baseline = reference_class(config, dataset).to(config["device"])
        with torch.no_grad():
            candidate_parameters = dict(model.named_parameters())
            for name, parameter in baseline.named_parameters():
                candidate_parameter = candidate_parameters.get(name)
                if (
                    candidate_parameter is not None
                    and candidate_parameter.shape == parameter.shape
                ):
                    parameter.copy_(candidate_parameter)
        user_count = int(dataset.user_num)
        item_count = int(dataset.item_num)
        if user_count < 3 or item_count < 4:
            raise AssertionError("behavioral probe dataset is too small")
        interaction = Interaction(
            {
                model.USER_ID: torch.tensor([1, 2], device=config["device"]),
                model.ITEM_ID: torch.tensor([1, 2], device=config["device"]),
                model.NEG_ITEM_ID: torch.tensor([2, 3], device=config["device"]),
            }
        )
        model.eval()
        baseline.eval()
        with torch.no_grad():
            candidate_score = model.predict(interaction)
            baseline_score = baseline.predict(interaction)
        score_delta = float(torch.max(torch.abs(candidate_score - baseline_score)).item())
        model.train()
        baseline.train()
        candidate_loss = model.calculate_loss(interaction)
        baseline_loss = baseline.calculate_loss(interaction)
        candidate_losses = candidate_loss if isinstance(candidate_loss, tuple) else (candidate_loss,)
        baseline_losses = baseline_loss if isinstance(baseline_loss, tuple) else (baseline_loss,)
        candidate_total = sum(item.reshape(()) for item in candidate_losses)
        baseline_total = sum(item.reshape(()) for item in baseline_losses)
        loss_delta = float(torch.abs(candidate_total.detach() - baseline_total.detach()).item())
        baseline_names = {name for name, _value in baseline.named_parameters()}
        extra_parameters = tuple(
            sorted(name for name, _value in model.named_parameters() if name not in baseline_names)
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        baseline_parameter_count = sum(parameter.numel() for parameter in baseline.parameters())
        if not math.isfinite(score_delta) or not math.isfinite(loss_delta):
            raise AssertionError("behavioral difference probe is non-finite")
        if score_delta <= 1e-7 and loss_delta <= 1e-7:
            raise AssertionError("candidate behavior is indistinguishable from inherited BPR")
        if (
            require_extra_parameters
            and not extra_parameters
            and parameter_count <= baseline_parameter_count
        ):
            raise AssertionError("candidate has no additional trainable mechanism")
        evidence.update(
            {
                "base_model_config": base_model_config,
                "baseline_parameter_count": baseline_parameter_count,
                "behavioral_loss_max_abs_delta": loss_delta,
                "behavioral_score_max_abs_delta": score_delta,
                "candidate_parameter_count": parameter_count,
                "extra_parameter_names": extra_parameters,
                "overridden_behavioral_methods": overridden,
                "probe_status": "PASS",
            }
        )

    return check


def _qualification_fixture(
    repo_root: Path,
    *,
    seed: int,
    root: Path,
    base_model_config: str = "BPR",
) -> RecBoleQualificationFixture:
    mini_data = (
        repo_root
        / "tests/experiments/helix_abc_v1/fixtures/innovation_spine/data"
    )
    return RecBoleQualificationFixture(
        project_root=repo_root,
        recbole_root=RECBole_ROOT,
        data_path=mini_data,
        dataset="mini",
        base_model_config=base_model_config,
        seed=seed,
        checkpoint_dir=root / "qualification/checkpoints",
        runtime_identity_ref="repo:docs/research_line/vnext/R1_R2_RUNTIME_DEPENDENCY_LOCK_V1#runtime_identity",
        runtime_identity_digest=MANIFEST_SHA256,
    )


def _materialize_and_qualify(
    *,
    repo_root: Path,
    side_root: Path,
    slot_id: str,
    seed: int,
    spec: Any,
    implementation: Mapping[str, Any],
    implementation_prompt_digest: str,
    tool_policy_digest: str,
    run_identity: str = CORRECTED_RUN_IDENTITY,
    policy: SharedImplementerPolicy | None = None,
    unit_check_factory: Callable[
        [dict[str, Any]], Callable[[Any, Any, Any], None]
    ] = _shared_behavioral_unit_check,
) -> tuple[MaterializedCandidate, MechanicalQualificationRun, dict[str, Any]]:
    policy = policy or _shared_policy(implementation_prompt_digest, tool_policy_digest)
    request = build_shared_implementer_request(spec, policy=policy)
    candidate_parent = side_root / "candidates" / slot_id
    candidate_parent.mkdir(parents=True, exist_ok=True)
    candidate_root = candidate_parent / str(request["blind_candidate_id"])
    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=implementation,
        candidate_root=candidate_root,
        candidate_root_ref=(
            f"{run_identity}-candidate-root:"
            f"{sha256_digest({'path': candidate_root.as_posix()})}"
        ),
    )
    behavior: dict[str, Any] = {}
    base_fixture = _qualification_fixture(repo_root, seed=seed, root=side_root / "qualification" / slot_id)
    fixture = RecBoleQualificationFixture(
        project_root=base_fixture.project_root,
        recbole_root=base_fixture.recbole_root,
        data_path=base_fixture.data_path,
        dataset=base_fixture.dataset,
        base_model_config=base_fixture.base_model_config,
        seed=base_fixture.seed,
        checkpoint_dir=base_fixture.checkpoint_dir,
        runtime_identity_ref=policy.runtime_identity_ref,
        runtime_identity_digest=policy.runtime_identity_digest,
    )
    qualification = MechanicalRecBoleAdapterV1().qualify(
        materialized.package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=fixture,
        unit_check=unit_check_factory(behavior),
    )
    return materialized, qualification, behavior


def _bpr_comparator_execution_recipe(
    *,
    spec: Any,
    run_id: str,
    entrypoint: str,
    source_sha256: str,
) -> dict[str, Any]:
    comparator_ref = f"{CORRECTED_RUN_IDENTITY}:bpr-comparator:{run_id}"
    comparator_digest = sha256_digest(
        {
            "base_model_config": "BPR",
            "entrypoint": entrypoint,
            "entrypoint_source_sha256": source_sha256,
            "model": "BPR",
        }
    )
    return canonical_value(
        {
            "base_model_config": "BPR",
            "capability_digest": comparator_digest,
            "capability_family": "BPR_MF",
            "capability_ref": comparator_ref,
            "comparator_digest": comparator_digest,
            "comparator_ref": comparator_ref,
            "config": {},
            "dataset": COMMON_DATASET,
            "entrypoint": entrypoint,
            "entrypoint_source_sha256": source_sha256,
            "evaluator": COMMON_EVALUATOR,
            "execution_role": "COMPARATOR",
            "mechanism_id": run_id,
            "model": "BPR",
            "profile_digest": spec.current_profile_digest,
            "profile_ref": spec.current_profile_ref,
            "split": COMMON_SPLIT,
        }
    )


def _fresh_r1_candidate_execution_recipe(
    *,
    spec: Any,
    capability: Any,
    package: Any,
    run_id: str,
    entrypoint: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Bind the current R1 qualified candidate's explicit BPR interface.

    R1 qualification currently requires a general pairwise BPR-compatible
    implementation.  That is an explicit caller recipe; it is not a runner
    default.  A later family-neutral producer can replace these two model
    fields without changing the worker seam.
    """

    return canonical_value(
        {
            "base_model_config": "BPR",
            "candidate_package_digest": package.digest,
            "candidate_package_ref": package.package_id,
            "candidate_root_digest": package.candidate_root_digest,
            "candidate_root_ref": package.candidate_root_ref,
            "candidate_source_tree_digest": package.source_tree_digest,
            "capability_digest": capability.digest,
            "capability_family": "BPR_MF",
            "capability_ref": capability.capability_id,
            "config": {},
            "dataset": COMMON_DATASET,
            "entrypoint": entrypoint,
            "entrypoint_source_sha256": source_sha256,
            "evaluator": COMMON_EVALUATOR,
            "execution_role": "CANDIDATE",
            "mechanism_id": run_id,
            "model": "BPR",
            "profile_digest": spec.current_profile_digest,
            "profile_ref": spec.current_profile_ref,
            "split": COMMON_SPLIT,
        }
    )


def _symlink_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), target, target_is_directory=source.is_dir())


def _write_start_gate(path: Path, identity: Mapping[str, Any]) -> None:
    _write_new_json(path, {**identity, "gate_status": "TRAINING_AUTHORIZED"})


def _round_test_feedback_metrics(
    worker: Mapping[str, Any],
) -> tuple[dict[str, float], bool]:
    identity_matches = (
        worker.get("metric_source") == "BEST_CHECKPOINT_TEST_RESULT"
        and worker.get("online_partition_role") == "ROUND_TEST_FEEDBACK"
    )
    payload = worker.get("test_result", {}) if identity_matches else {}
    metrics = {
        str(key).lower(): float(value)
        for key, value in dict(payload).items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    return metrics, identity_matches


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _flatten_numeric_values(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_flatten_numeric_values(item))
        return values
    converted = _finite_number(value)
    return [] if converted is None else [converted]


def _loss_observations(telemetry: Mapping[str, Any]) -> tuple[list[float], bool]:
    """Return scalar loss observations and whether any explicit value is invalid."""

    observations: list[float] = []
    invalid = telemetry.get("non_finite_loss") is True
    trend = telemetry.get("loss_trend")
    if isinstance(trend, (list, tuple)):
        for item in trend:
            flattened = _flatten_numeric_values(item)
            if flattened:
                observations.append(sum(flattened) / len(flattened))
            elif item is not None:
                invalid = True
    phases = telemetry.get("phase_records")
    if isinstance(phases, (list, tuple)):
        for row in phases:
            if not isinstance(row, Mapping) or row.get("phase") != "TRAIN":
                continue
            value = row.get("loss")
            flattened = _flatten_numeric_values(value)
            if flattened and not observations:
                observations.append(sum(flattened) / len(flattened))
            elif value is not None and not flattened:
                invalid = True
    active = telemetry.get("active_progress")
    if isinstance(active, Mapping):
        value = active.get("last_loss_observation")
        flattened = _flatten_numeric_values(value)
        if flattened:
            observed = sum(flattened) / len(flattened)
            if not observations or observed != observations[-1]:
                observations.append(observed)
        elif value is not None:
            invalid = True
    return observations, invalid


def _phase_rows(telemetry: Mapping[str, Any], phase: str) -> list[Mapping[str, Any]]:
    rows = telemetry.get("phase_records")
    if not isinstance(rows, (list, tuple)):
        return []
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("phase") == phase
        and row.get("status") not in {"RUNTIME_FAILURE", "FAILED"}
    ]


def _completed_epochs(telemetry: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]) -> int:
    explicit = telemetry.get("epochs_completed")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    return sum(
        row.get("status") in {None, "SUCCESS", "PHASE_COMPLETED"}
        for row in train_rows
    )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _prediction_number(
    prediction: Mapping[str, Any] | None,
    *keys: str,
) -> float | None:
    if not isinstance(prediction, Mapping):
        return None
    containers: list[Mapping[str, Any]] = [prediction]
    decision_inputs = prediction.get("decision_inputs")
    if isinstance(decision_inputs, Mapping):
        containers.append(decision_inputs)
    for container in containers:
        for key in keys:
            if key not in container:
                continue
            raw = container.get(key)
            if raw is None:
                continue
            value = _finite_number(raw)
            if value is None or value <= 0:
                raise FreshR1Error(f"resource prediction {key} must be positive")
            return value
    return None


def resolve_candidate_deadline_seconds(
    *,
    default_seconds: int,
    prediction: Mapping[str, Any] | None = None,
    final_worker_ceiling_seconds: int = MAX_WORKER_CEILING_SECONDS,
) -> int:
    """Resolve a candidate allocation while retaining a separate hard ceiling."""

    if (
        isinstance(default_seconds, bool)
        or not isinstance(default_seconds, int)
        or default_seconds < 1
    ):
        raise FreshR1Error("default candidate deadline must be a positive integer")
    if (
        isinstance(final_worker_ceiling_seconds, bool)
        or not isinstance(final_worker_ceiling_seconds, int)
        or final_worker_ceiling_seconds < 1
    ):
        raise FreshR1Error("final worker ceiling must be a positive integer")
    hard_ceiling = min(MAX_WORKER_CEILING_SECONDS, final_worker_ceiling_seconds)
    predicted = _prediction_number(
        prediction,
        "candidate_deadline_seconds",
        "requested_deadline_seconds",
        "deadline_seconds",
    )
    return max(1, min(hard_ceiling, math.ceil(predicted or default_seconds)))


def _first_progress_deadline_seconds(
    *,
    candidate_deadline_seconds: int,
    prediction: Mapping[str, Any] | None,
) -> float:
    predicted = _prediction_number(
        prediction,
        "first_progress_deadline_seconds",
        "first_epoch_deadline_seconds",
        "initialization_deadline_seconds",
    )
    if predicted is not None:
        return min(float(candidate_deadline_seconds), predicted)
    predicted_epoch = _prediction_number(
        prediction,
        "epoch_seconds",
        "train_epoch_seconds",
        "estimated_epoch_seconds",
    )
    if predicted_epoch is not None:
        return min(
            float(candidate_deadline_seconds),
            max(30.0, 2.0 * predicted_epoch),
        )
    return float(candidate_deadline_seconds)


def _prediction_cycle_upper_seconds(prediction: Mapping[str, Any] | None) -> float | None:
    if not isinstance(prediction, Mapping):
        return None
    interval = prediction.get("prediction_interval_seconds")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        upper = _finite_number(interval[1])
        if upper is not None and upper > 0:
            return upper
    return _prediction_number(
        prediction,
        "epoch_upper_seconds",
        "epoch_p90_seconds",
        "cycle_upper_seconds",
    )


def _observed_cycle_seconds(
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    prediction: Mapping[str, Any] | None,
) -> float | None:
    train = [
        float(row["wall_time_ms"]) / 1000.0
        for row in train_rows
        if _finite_number(row.get("wall_time_ms")) is not None
        and float(row["wall_time_ms"]) > 0
    ]
    evaluation = [
        float(row["wall_time_ms"]) / 1000.0
        for row in eval_rows
        if _finite_number(row.get("wall_time_ms")) is not None
        and float(row["wall_time_ms"]) > 0
    ]
    observed = _median(train) or _median(evaluation)
    if train and evaluation:
        observed = _median(
            [
                (train[index] if index < len(train) else train[-1])
                + (evaluation[index] if index < len(evaluation) else evaluation[-1])
                for index in range(max(len(train), len(evaluation)))
            ]
        )
    if observed is not None:
        return observed
    predicted = _prediction_number(
        prediction,
        "cycle_seconds",
        "epoch_seconds",
        "train_epoch_seconds",
    )
    if predicted is not None:
        evaluation_seconds = _prediction_number(
            prediction,
            "eval_seconds",
            "evaluation_seconds",
        )
        return predicted + (evaluation_seconds or 0.0)
    return None


def _native_stop_projection_seconds(
    *,
    telemetry: Mapping[str, Any],
    completed_epochs: int,
    elapsed_seconds: float,
    cycle_seconds: float | None,
    epochs_requested: int,
    prediction: Mapping[str, Any] | None,
) -> float | None:
    if cycle_seconds is None:
        return _prediction_number(
            prediction,
            "native_seal_seconds",
            "estimated_total_wall_time_seconds",
        )
    full_projection = elapsed_seconds + max(0, epochs_requested - completed_epochs) * cycle_seconds
    best_epoch = telemetry.get("best_observed_epoch")
    if not isinstance(best_epoch, int) or best_epoch < 0:
        eval_rows = _phase_rows(telemetry, "EVAL")
        scored = [
            row
            for row in eval_rows
            if _finite_number(row.get("valid_score")) is not None
        ]
        if scored:
            best_epoch = max(
                scored,
                key=lambda row: float(row["valid_score"]),
            ).get("epoch")
    patience = _prediction_number(
        prediction,
        "native_stopping_patience",
        "stopping_step",
    )
    patience_epochs = int(patience or DEFAULT_NATIVE_STOPPING_PATIENCE)
    if isinstance(best_epoch, int) and completed_epochs >= best_epoch:
        epochs_since_best = completed_epochs - best_epoch
        remaining = max(1, patience_epochs - epochs_since_best)
        native_projection = elapsed_seconds + remaining * cycle_seconds
        return min(full_projection, native_projection)
    return full_projection


def _failure_scope_for_trigger(trigger: str | None) -> str | None:
    if trigger is None:
        return None
    if trigger in {"SHARED_INFRASTRUCTURE", "DATASET_OR_EVALUATOR_UNAVAILABLE"}:
        return "SHARED_INFRASTRUCTURE"
    if trigger in {"WORKER_TRANSIENT", "WORKER_EXIT"}:
        return "WORKER_TRANSIENT"
    if trigger in {
        "MEMORY_INFEASIBLE",
        "NO_FIRST_PROGRESS",
        "EPOCH_EVAL_PREDICTION",
        "PROJECTED_SEAL_EXCEEDS_FINAL_CEILING",
        "HARD_CEILING",
    }:
        return "LINEAGE_COMPUTE_PATTERN"
    return "CANDIDATE_LOCAL"


def _decision(
    *,
    action: str,
    state: str,
    trigger: str | None,
    reason: str,
    elapsed_seconds: float,
    candidate_deadline_seconds: int,
    final_worker_ceiling_seconds: int,
    projected_seal_seconds: float | None = None,
    failure_scope: str | None = None,
) -> dict[str, Any]:
    return canonical_value(
        {
            "action": action,
            "candidate_deadline_seconds": candidate_deadline_seconds,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
            "failure_scope": failure_scope or _failure_scope_for_trigger(trigger),
            "final_worker_ceiling_seconds": final_worker_ceiling_seconds,
            "health_state": state,
            "mechanism_effect_update_allowed": action
            not in {HEALTH_ACTION_HEALTH_ABORT, HEALTH_ACTION_HARD_CEILING},
            "projected_seal_seconds": projected_seal_seconds,
            "reason": reason,
            "schema": "recclaw.training-health-supervisor.v1",
            "trigger": trigger,
        }
    )


def assess_training_health(
    telemetry: Mapping[str, Any] | None,
    *,
    elapsed_seconds: float,
    candidate_deadline_seconds: int,
    final_worker_ceiling_seconds: int = MAX_WORKER_CEILING_SECONDS,
    prediction: Mapping[str, Any] | None = None,
    process_return_code: int | None = None,
    telemetry_enabled: bool = True,
    epochs_requested: int = EXPERIMENT_EPOCHS,
) -> dict[str, Any]:
    """Classify one durable telemetry snapshot without censoring finite science."""

    if elapsed_seconds < 0:
        raise FreshR1Error("health elapsed_seconds must not be negative")
    if (
        isinstance(candidate_deadline_seconds, bool)
        or not isinstance(candidate_deadline_seconds, int)
        or candidate_deadline_seconds < 1
    ):
        raise FreshR1Error("candidate deadline must be a positive integer")
    if (
        isinstance(final_worker_ceiling_seconds, bool)
        or not isinstance(final_worker_ceiling_seconds, int)
        or final_worker_ceiling_seconds < 1
    ):
        raise FreshR1Error("final worker ceiling must be a positive integer")
    if (
        isinstance(epochs_requested, bool)
        or not isinstance(epochs_requested, int)
        or epochs_requested < 1
    ):
        raise FreshR1Error("epochs_requested must be a positive integer")
    hard_ceiling = min(MAX_WORKER_CEILING_SECONDS, final_worker_ceiling_seconds)
    candidate_deadline = min(candidate_deadline_seconds, hard_ceiling)
    current = telemetry if isinstance(telemetry, Mapping) else {}
    active = current.get("active_progress")
    active = active if isinstance(active, Mapping) else {}
    train_rows = _phase_rows(current, "TRAIN")
    eval_rows = _phase_rows(current, "EVAL")
    completed = _completed_epochs(current, train_rows)
    phase = str(active.get("phase") or "")
    state = (
        HEALTH_STATE_VALIDATION_PROGRESS
        if phase == "EVAL" or eval_rows
        else HEALTH_STATE_TRAIN_PROGRESS
        if phase == "TRAIN" or completed > 0
        else HEALTH_STATE_INITIALIZED
        if current
        else HEALTH_STATE_STARTING
    )

    if process_return_code == 0:
        return _decision(
            action=HEALTH_ACTION_NATIVE_SEAL,
            state=HEALTH_STATE_NATIVE_SEAL,
            trigger=None,
            reason="worker completed and native result is available",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )
    if process_return_code is not None and process_return_code != 0:
        explicit_scope = current.get("failure_scope")
        trigger = (
            "SHARED_INFRASTRUCTURE"
            if current.get("shared_infrastructure_failure") is True
            else "WORKER_TRANSIENT"
        )
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger=trigger,
            reason="worker exited before a valid native seal",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
            failure_scope=(
                str(explicit_scope)
                if explicit_scope in {
                    "CANDIDATE_LOCAL",
                    "LINEAGE_COMPUTE_PATTERN",
                    "WORKER_TRANSIENT",
                    "SHARED_INFRASTRUCTURE",
                }
                else None
            ),
        )

    if current.get("shared_infrastructure_failure") is True or current.get(
        "failure_scope"
    ) == "SHARED_INFRASTRUCTURE":
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="SHARED_INFRASTRUCTURE",
            reason="durable telemetry marked a shared infrastructure failure",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
            failure_scope="SHARED_INFRASTRUCTURE",
        )
    if current.get("worker_transient_failure") is True:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="WORKER_TRANSIENT",
            reason="durable telemetry marked a recoverable worker failure",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
            failure_scope="WORKER_TRANSIENT",
        )

    losses, invalid_loss = _loss_observations(current)
    if invalid_loss or current.get("loss_is_finite") is False:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="NON_FINITE_LOSS",
            reason="loss telemetry contains a non-finite observation",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )
    if len(losses) >= 3:
        recent = losses[-3:]
        if (
            all(recent[index] >= recent[index - 1] for index in range(1, len(recent)))
            and recent[0] > 0
            and recent[-1] / recent[0] >= 8.0
            and min(
                recent[index] / recent[index - 1]
                for index in range(1, len(recent))
                if recent[index - 1] > 0
            )
            >= 1.5
        ):
            return _decision(
                action=HEALTH_ACTION_HEALTH_ABORT,
                state=HEALTH_STATE_HEALTH_ABORT,
                trigger="EXPLOSIVE_LOSS",
                reason="finite loss has a robust multi-step explosive trend",
                elapsed_seconds=elapsed_seconds,
                candidate_deadline_seconds=candidate_deadline,
                final_worker_ceiling_seconds=hard_ceiling,
            )

    optimizer = current.get("optimizer")
    optimizer = optimizer if isinstance(optimizer, Mapping) else {}
    no_op = current.get("optimizer_no_op") is True or optimizer.get("no_op") is True
    expected_steps = current.get("expected_optimizer_steps")
    actual_steps = current.get("optimizer_steps")
    if (
        isinstance(expected_steps, int)
        and not isinstance(expected_steps, bool)
        and expected_steps > 0
        and isinstance(actual_steps, int)
        and not isinstance(actual_steps, bool)
        and actual_steps == 0
    ):
        no_op = True
    update_records = current.get("update_records")
    if isinstance(update_records, (list, tuple)) and update_records:
        numeric_updates = [
            _finite_number(
                row.get("parameter_update_norm", row.get("update_norm"))
            )
            for row in update_records
            if isinstance(row, Mapping)
        ]
        if numeric_updates and all(value is not None and value <= 1e-12 for value in numeric_updates):
            no_op = True
    if no_op:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="OPTIMIZER_NO_OP",
            reason="durable optimizer telemetry proves zero parameter updates",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    predicted_memory = _prediction_number(
        prediction,
        "peak_memory_prediction_mib",
        "predicted_peak_memory_mib",
    )
    observed_memory = _finite_number(current.get("peak_gpu_memory_mib"))
    memory_limit = _prediction_number(
        prediction,
        "gpu_memory_limit_mib",
        "memory_limit_mib",
    )
    if memory_limit is None:
        memory_limit = _finite_number(current.get("gpu_memory_limit_mib"))
    memory_infeasible = current.get("memory_infeasible") is True
    if memory_limit is not None:
        memory_infeasible = memory_infeasible or (
            predicted_memory is not None
            and predicted_memory >= memory_limit * 0.95
        ) or (
            observed_memory is not None
            and observed_memory >= memory_limit * 0.99
        )
    if memory_infeasible:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="MEMORY_INFEASIBLE",
            reason="predicted or observed memory exceeds the admitted capacity margin",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    active_phase_elapsed = _finite_number(active.get("phase_elapsed_ms"))
    cycle_upper = _prediction_cycle_upper_seconds(prediction)
    if (
        active_phase_elapsed is not None
        and cycle_upper is not None
        and active_phase_elapsed / 1000.0 > max(60.0, cycle_upper * 2.0)
    ):
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="EPOCH_EVAL_PREDICTION",
            reason="active train/eval phase exceeds the prediction envelope",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    if not telemetry_enabled:
        if elapsed_seconds >= hard_ceiling:
            return _decision(
                action=HEALTH_ACTION_HARD_CEILING,
                state=HEALTH_STATE_HARD_CEILING,
                trigger="HARD_CEILING",
                reason="worker reached the final safety ceiling without telemetry",
                elapsed_seconds=elapsed_seconds,
                candidate_deadline_seconds=candidate_deadline,
                final_worker_ceiling_seconds=hard_ceiling,
            )
        return _decision(
            action=HEALTH_ACTION_CONTINUE,
            state=state,
            trigger=None,
            reason="telemetry is disabled; no health inference is permitted",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    if elapsed_seconds >= hard_ceiling:
        return _decision(
            action=HEALTH_ACTION_HARD_CEILING,
            state=HEALTH_STATE_HARD_CEILING,
            trigger="HARD_CEILING",
            reason="worker reached the final 3600-second safety ceiling",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    first_progress_deadline = _first_progress_deadline_seconds(
        candidate_deadline_seconds=candidate_deadline,
        prediction=prediction,
    )
    if not current and elapsed_seconds >= candidate_deadline:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="WORKER_TRANSIENT",
            reason="durable telemetry never became readable before candidate deadline",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
            failure_scope="WORKER_TRANSIENT",
        )
    if completed == 0 and elapsed_seconds >= first_progress_deadline:
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="NO_FIRST_PROGRESS",
            reason="no completed training epoch arrived by the predicted first-progress deadline",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
        )

    cycle_seconds = _observed_cycle_seconds(
        train_rows,
        eval_rows,
        prediction=prediction,
    )
    projected_seal = _native_stop_projection_seconds(
        telemetry=current,
        completed_epochs=completed,
        elapsed_seconds=elapsed_seconds,
        cycle_seconds=cycle_seconds,
        epochs_requested=epochs_requested,
        prediction=prediction,
    )
    if elapsed_seconds >= candidate_deadline:
        if projected_seal is not None and projected_seal <= hard_ceiling:
            return _decision(
                action=HEALTH_ACTION_CONTINUE,
                state=state,
                trigger=None,
                reason=(
                    "candidate allocation elapsed, but projected native seal fits "
                    "inside the final ceiling"
                ),
                elapsed_seconds=elapsed_seconds,
                candidate_deadline_seconds=candidate_deadline,
                final_worker_ceiling_seconds=hard_ceiling,
                projected_seal_seconds=projected_seal,
            )
        return _decision(
            action=HEALTH_ACTION_HEALTH_ABORT,
            state=HEALTH_STATE_HEALTH_ABORT,
            trigger="PROJECTED_SEAL_EXCEEDS_FINAL_CEILING",
            reason="projected full/native seal cannot fit inside the final ceiling",
            elapsed_seconds=elapsed_seconds,
            candidate_deadline_seconds=candidate_deadline,
            final_worker_ceiling_seconds=hard_ceiling,
            projected_seal_seconds=projected_seal,
        )

    return _decision(
        action=HEALTH_ACTION_CONTINUE,
        state=state,
        trigger=None,
        reason="finite training is making admissible progress; native stopping remains authoritative",
        elapsed_seconds=elapsed_seconds,
        candidate_deadline_seconds=candidate_deadline,
        final_worker_ceiling_seconds=hard_ceiling,
        projected_seal_seconds=projected_seal,
    )


def classify_failure_scope(
    *,
    trigger: str | None,
    telemetry: Mapping[str, Any] | None = None,
    worker: Mapping[str, Any] | None = None,
) -> str | None:
    """Assign the narrowest supported scope for routing/resource memory."""

    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    worker = worker if isinstance(worker, Mapping) else {}
    explicit = telemetry.get("failure_scope") or worker.get("failure_scope")
    if explicit in {
        "CANDIDATE_LOCAL",
        "LINEAGE_COMPUTE_PATTERN",
        "WORKER_TRANSIENT",
        "SHARED_INFRASTRUCTURE",
    }:
        return str(explicit)
    text = " ".join(
        str(worker.get(key) or "")
        for key in ("error_message", "error_type", "traceback")
    ).lower()
    if any(
        marker in text
        for marker in (
            "cuda_device_capability_unavailable",
            "cuda unavailable",
            "nvidia driver",
            "nccl",
            "dataset",
            "heldout",
            "evaluator",
            "recbole source",
            "no space left",
            "filesystem",
        )
    ):
        return "SHARED_INFRASTRUCTURE"
    if any(
        marker in text
        for marker in (
            "brokenpipe",
            "connectionreset",
            "connectionerror",
            "worker exited",
            "process group",
            "timed out",
        )
    ):
        return "WORKER_TRANSIENT"
    return _failure_scope_for_trigger(trigger)


def _single_cuda_visible_device(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise FreshR1Error(
            "cuda_visible_devices must be one normalized physical device token"
        )
    tokens = tuple(token.strip() for token in value.split(","))
    if len(tokens) != 1 or not tokens[0] or tokens[0] in {"-1", "NoDevFiles"}:
        raise FreshR1Error(
            "cuda_visible_devices must bind exactly one concrete device"
        )
    return tokens[0]


def _validated_gpu_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FreshR1Error("gpu_id must be a non-negative integer")
    return int(value)


def _worker_recipe_for_gpu_selection(
    worker_recipe: Mapping[str, Any],
    gpu_id: int | None,
) -> Mapping[str, Any]:
    """Add only direct-device fields to the worker-facing recipe."""

    validated_gpu_id = _validated_gpu_id(gpu_id)
    if validated_gpu_id is None:
        return worker_recipe
    config = worker_recipe.get("config", {})
    if not isinstance(config, Mapping):
        raise FreshR1Error("worker recipe config must be a mapping")
    return canonical_value(
        {
            **dict(worker_recipe),
            "config": {
                **dict(config),
                "gpu_id": validated_gpu_id,
                "worker": 8,
            },
            "gpu_selection_mode": DIRECT_GPU_SELECTION_MODE,
        }
    )


def _reservation_text(identity: Mapping[str, Any], field_name: str) -> str:
    value = identity.get(field_name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise FreshR1Error(
            f"GPU reservation identity {field_name} must be normalized and non-empty"
        )
    return value


def validate_gpu_reservation_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    cuda_visible_devices: str | None,
    physical_gpu_selector: str | None = None,
    run_id: str | None = None,
) -> Mapping[str, Any] | None:
    """Validate an optional sealed, single-device reservation identity.

    This is evidence input only.  It does not allocate or renew a device lease;
    without the caller-supplied sealed identity the launcher reports reserved
    GPU worker seconds as unmeasured.
    """

    visible = _single_cuda_visible_device(cuda_visible_devices)
    physical_selector = _single_cuda_visible_device(physical_gpu_selector)
    if visible is not None and physical_selector is not None:
        raise FreshR1Error(
            "cuda_visible_devices and physical_gpu_selector are mutually exclusive"
        )
    selector = visible if visible is not None else physical_selector
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise FreshR1Error("gpu_reservation_evidence must be a mapping")
    if evidence.get("schema") != GPU_RESERVATION_EVIDENCE_SCHEMA:
        raise FreshR1Error("GPU reservation evidence schema is unsupported")
    reservation_ref = evidence.get("reservation_ref")
    if (
        not isinstance(reservation_ref, str)
        or not reservation_ref
        or reservation_ref != reservation_ref.strip()
    ):
        raise FreshR1Error("GPU reservation evidence requires reservation_ref")
    identity = evidence.get("identity")
    if not isinstance(identity, Mapping):
        raise FreshR1Error("GPU reservation evidence requires an identity mapping")
    host = _reservation_text(identity, "host")
    physical_gpu_id = _reservation_text(identity, "physical_gpu_id")
    identity_visible = _reservation_text(identity, "cuda_visible_devices")
    if selector is None or identity_visible != selector:
        raise FreshR1Error(
            "GPU reservation identity must match the selected physical GPU"
        )
    if physical_gpu_id != selector:
        raise FreshR1Error(
            "GPU reservation physical_gpu_id must equal the selected physical GPU"
        )
    reservation_owner_ref = _reservation_text(
        identity,
        "reservation_owner_ref",
    )
    if run_id is not None:
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id != run_id.strip()
        ):
            raise FreshR1Error("run_id must be normalized and non-empty")
        if reservation_owner_ref != run_id:
            raise FreshR1Error(
                "GPU reservation owner does not match the training run_id"
            )
    observed_at_utc = _reservation_text(identity, "observed_at_utc")
    if not observed_at_utc.endswith("Z"):
        raise FreshR1Error("GPU reservation observed_at_utc must be UTC")
    try:
        parsed_observed_at = datetime.fromisoformat(
            observed_at_utc[:-1] + "+00:00"
        )
    except ValueError as error:
        raise FreshR1Error(
            "GPU reservation observed_at_utc must be normalized ISO-8601 UTC"
        ) from error
    if (
        parsed_observed_at.tzinfo is None
        or parsed_observed_at.utcoffset() is None
        or parsed_observed_at.utcoffset().total_seconds() != 0
        or parsed_observed_at.isoformat().replace("+00:00", "Z")
        != observed_at_utc
    ):
        raise FreshR1Error(
            "GPU reservation observed_at_utc must be normalized ISO-8601 UTC"
        )
    for field_name in ("device_inventory_sha256", "process_snapshot_sha256"):
        try:
            validate_sha256(
                identity.get(field_name),
                field_name=f"gpu_reservation_evidence.identity.{field_name}",
            )
        except (TypeError, ValueError) as error:
            raise FreshR1Error(str(error)) from error
    if identity.get("exclusive") is not True:
        raise FreshR1Error("GPU reservation evidence must prove exclusivity")
    scope = _reservation_text(identity, "scope")
    if scope not in _GPU_RESERVATION_SCOPES:
        raise FreshR1Error("GPU reservation evidence scope is not candidate-local")
    device_identity_fields = [
        field_name
        for field_name in ("device_uuid", "device_ref")
        if identity.get(field_name) is not None
    ]
    if not device_identity_fields:
        raise FreshR1Error(
            "GPU reservation evidence requires a concrete device_uuid or device_ref"
        )
    for field_name in device_identity_fields:
        _reservation_text(identity, field_name)
    try:
        identity_digest = validate_sha256(
            evidence.get("identity_digest"),
            field_name="gpu_reservation_evidence.identity_digest",
        )
        reservation_digest = validate_sha256(
            evidence.get("reservation_digest"),
            field_name="gpu_reservation_evidence.reservation_digest",
        )
    except (TypeError, ValueError) as error:
        raise FreshR1Error(str(error)) from error
    if identity_digest != sha256_digest(identity):
        raise FreshR1Error("GPU reservation identity digest mismatch")
    sealed = {
        "identity": canonical_value(dict(identity)),
        "identity_digest": identity_digest,
        "reservation_ref": reservation_ref,
        "schema": GPU_RESERVATION_EVIDENCE_SCHEMA,
    }
    if reservation_digest != sha256_digest(sealed):
        raise FreshR1Error("GPU reservation sealed digest mismatch")
    if host != socket.gethostname():
        raise FreshR1Error("GPU reservation host does not match the current host")
    return canonical_value(dict(evidence))


def _cross_check_training_device_evidence(
    device_evidence: Mapping[str, Any] | None,
    *,
    cuda_visible_devices: str | None,
    gpu_id: int | None = None,
    reservation_evidence: Mapping[str, Any] | None,
) -> str | None:
    """Return a contradiction without inventing physical identity evidence."""

    if not isinstance(device_evidence, Mapping):
        return None
    visible = _single_cuda_visible_device(cuda_visible_devices)
    direct_selector = str(gpu_id) if gpu_id is not None else None
    binding_present = (
        visible is not None
        or direct_selector is not None
        or reservation_evidence is not None
    )
    if gpu_id is not None:
        if device_evidence.get("selection_mode") != DIRECT_GPU_SELECTION_MODE:
            return "worker did not report direct RecBole gpu_id selection"
        if device_evidence.get("gpu_id") != gpu_id:
            return "worker gpu_id differs from the direct launch selector"
        if str(device_evidence.get("physical_gpu_id")) != direct_selector:
            return "worker physical_gpu_id differs from the direct launch selector"
        if device_evidence.get("cuda_visible_devices") is not None:
            return "direct gpu_id worker unexpectedly reports CUDA_VISIBLE_DEVICES"
    if "cuda_available" in device_evidence and device_evidence.get(
        "cuda_available"
    ) is not True:
        if binding_present:
            return "worker reports CUDA unavailable"
        return None
    device_count = device_evidence.get("cuda_device_count")
    if (visible is not None or gpu_id is not None) and device_count is not None:
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            return "worker cuda_device_count is not an integer"
        if device_count != 1:
            return "single-device binding contradicts worker cuda_device_count"
    for field_name in ("logical_device", "cuda_device_index", "current_device"):
        if (
            binding_present
            and field_name in device_evidence
            and device_evidence.get(field_name) != 0
        ):
            return f"worker {field_name} contradicts logical device 0"
    for field_name in ("cuda_visible_devices", "visible_device"):
        worker_visible = device_evidence.get(field_name)
        if worker_visible is not None and visible is not None:
            try:
                worker_visible = _single_cuda_visible_device(worker_visible)
            except FreshR1Error as error:
                return str(error)
            if worker_visible != visible:
                return f"worker {field_name} contradicts CUDA_VISIBLE_DEVICES"
    if reservation_evidence is None:
        return None
    identity = reservation_evidence.get("identity")
    if not isinstance(identity, Mapping):
        return "validated reservation identity is unavailable"
    for field_name in ("physical_gpu_id", "device_uuid", "device_ref"):
        worker_value = device_evidence.get(field_name)
        reserved_value = identity.get(field_name)
        if worker_value is not None and reserved_value is not None:
            if str(worker_value) != str(reserved_value):
                return f"worker {field_name} contradicts reservation identity"
    for worker_field, reservation_field in (
        ("cuda_device_name", "device_name"),
        ("gpu_name", "device_name"),
    ):
        worker_value = device_evidence.get(worker_field)
        reserved_value = identity.get(reservation_field)
        if worker_value is not None and reserved_value is not None:
            if str(worker_value) != str(reserved_value):
                return f"worker {worker_field} contradicts reservation identity"
    return None


def _parent_process_interval(
    started_ns: int,
    *,
    ended_ns: int | None = None,
) -> dict[str, Any]:
    if isinstance(started_ns, bool) or not isinstance(started_ns, int):
        raise FreshR1Error("parent process start monotonic timestamp is invalid")
    observed_end_ns = time.monotonic_ns() if ended_ns is None else ended_ns
    if isinstance(observed_end_ns, bool) or not isinstance(observed_end_ns, int):
        raise FreshR1Error("parent process end monotonic timestamp is invalid")
    if observed_end_ns < started_ns:
        raise FreshR1Error("parent process monotonic interval is inverted")
    elapsed_ns = observed_end_ns - started_ns
    return {
        "parent_process_started_monotonic_ns": started_ns,
        "parent_process_ended_monotonic_ns": observed_end_ns,
        "parent_process_interval_seconds": elapsed_ns / 1_000_000_000,
        "parent_process_interval_wall_time_ms": max(1, elapsed_ns // 1_000_000),
    }


def _finalize_process_outcome(
    value: Mapping[str, Any],
    *,
    started_ns: int,
    ended_ns: int | None,
    cuda_visible_devices: str | None,
    gpu_id: int | None,
    reservation_evidence: Mapping[str, Any] | None,
    device_evidence_consistent: bool = True,
) -> dict[str, Any]:
    interval = _parent_process_interval(started_ns, ended_ns=ended_ns)
    result = dict(value)
    result["parent_process_interval"] = interval
    result["wall_time_ms"] = interval["parent_process_interval_wall_time_ms"]
    visible = _single_cuda_visible_device(cuda_visible_devices)
    physical_selector = str(gpu_id) if gpu_id is not None else visible
    reservation_proven = (
        reservation_evidence is not None
        and physical_selector is not None
        and device_evidence_consistent
    )
    result["cuda_visible_devices"] = visible
    result["gpu_reservation_evidence"] = reservation_evidence
    result["gpu_reservation_status"] = (
        GPU_RESERVATION_STATUS_MEASURED
        if reservation_proven
        else (
            GPU_RESERVATION_STATUS_CONTRADICTORY
            if not device_evidence_consistent
            else GPU_RESERVATION_STATUS_UNMEASURED
        )
    )
    result["reserved_gpu_worker_seconds_semantics"] = (
        GPU_WORKER_SECONDS_SEMANTICS
    )
    result["reserved_gpu_worker_seconds"] = (
        interval["parent_process_interval_seconds"]
        if reservation_proven
        else None
    )
    if gpu_id is not None:
        result.update(
            {
                "gpu_id": gpu_id,
                "physical_gpu_id": physical_selector,
                "selection_mode": DIRECT_GPU_SELECTION_MODE,
            }
        )
    return canonical_value(result)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _supervise_training_process(
    process: subprocess.Popen[str],
    *,
    telemetry_path: Path,
    candidate_deadline_seconds: int,
    final_worker_ceiling_seconds: int,
    prediction: Mapping[str, Any] | None,
    telemetry_enabled: bool,
    epochs_requested: int,
    poll_interval_seconds: float = 0.25,
) -> tuple[str, str, int, dict[str, Any], Mapping[str, Any] | None, bool]:
    """Poll durable worker telemetry and terminate only typed health failures."""

    if poll_interval_seconds <= 0:
        raise FreshR1Error("supervisor poll interval must be positive")
    started = time.monotonic()
    last_telemetry: Mapping[str, Any] | None = None
    active_phase_key: tuple[object, object] | None = None
    active_phase_started = 0.0
    hard_ceiling = min(MAX_WORKER_CEILING_SECONDS, final_worker_ceiling_seconds)
    while True:
        current = _read_optional_json(telemetry_path) if telemetry_enabled else None
        if current is not None:
            active = current.get("active_progress")
            active = active if isinstance(active, Mapping) else {}
            phase_key = (active.get("phase"), active.get("epoch"))
            elapsed = time.monotonic() - started
            if phase_key != active_phase_key:
                active_phase_key = phase_key
                active_phase_started = elapsed
            if active.get("phase") is not None:
                current = {
                    **current,
                    "active_progress": {
                        **dict(active),
                        "phase_elapsed_ms": max(
                            0,
                            int((elapsed - active_phase_started) * 1000),
                        ),
                    },
                }
            last_telemetry = current
        return_code = process.poll()
        elapsed = time.monotonic() - started
        decision = assess_training_health(
            last_telemetry,
            elapsed_seconds=elapsed,
            candidate_deadline_seconds=candidate_deadline_seconds,
            final_worker_ceiling_seconds=hard_ceiling,
            prediction=prediction,
            process_return_code=return_code,
            telemetry_enabled=telemetry_enabled,
            epochs_requested=epochs_requested,
        )
        if return_code is not None:
            stdout, stderr = process.communicate()
            return (
                stdout or "",
                stderr or "",
                int(return_code),
                decision,
                last_telemetry,
                False,
            )
        if decision["action"] in {
            HEALTH_ACTION_HEALTH_ABORT,
            HEALTH_ACTION_HARD_CEILING,
        }:
            process.kill()
            stdout, stderr = process.communicate()
            return (
                stdout or "",
                stderr or "",
                124,
                decision,
                last_telemetry,
                True,
            )
        time.sleep(min(poll_interval_seconds, 0.25))


def _worker_environment(
    capability: Any,
    candidate_root: Path | None,
    cuda_visible_devices: str | None = None,
    gpu_id: int | None = None,
) -> dict[str, str]:
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "TZ",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if gpu_id is None and cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    environment.update(capability.environment)
    if gpu_id is not None:
        # Direct mode is selected by RecBole's explicit gpu_id config.  Remove
        # any inherited CVD so the launcher cannot silently become the selector.
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    roots = []
    if candidate_root is not None:
        roots.append(candidate_root.as_posix())
    roots.extend(
        [
            str(Path(__file__).resolve().parents[4] / "src"),
            str(Path(__file__).resolve().parents[4]),
            str(RECBole_ROOT.resolve()),
        ]
    )
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def run_development_training(
    *,
    repo_root: Path,
    side_root: Path,
    run_id: str,
    seed: int,
    candidate_root: Path | None,
    entrypoint: str,
    source_sha256: str,
    run_identity: str = CORRECTED_RUN_IDENTITY,
    authority: str = "user-delegated-corrected-formal-fresh-r1",
    timeout_seconds: int = 1500,
    recbole_commit_identity: str | None = None,
    expected_recbole_source_tree_digest: str | None = None,
    epochs: int = EXPERIMENT_EPOCHS,
    execution_purpose: str = "DEVELOPMENT_PILOT_OFFLINE_TOPN",
    resource_telemetry: bool = False,
    watchdog_seconds: int | None = None,
    final_worker_ceiling_seconds: int | None = None,
    resource_prediction: Mapping[str, Any] | None = None,
    prefix_contract_path: Path | None = None,
    execution_recipe: Mapping[str, Any] | None = None,
    cuda_visible_devices: str | None = None,
    gpu_id: int | None = None,
    gpu_reservation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if execution_recipe is None:
        raise FreshR1Error(
            "explicit execution_recipe is required; generic BPR execution is disabled"
        )
    try:
        validate_execution_recipe(execution_recipe)
    except ExperimentBindingError as error:
        raise FreshR1Error(str(error)) from error
    if execution_recipe["entrypoint"] != entrypoint:
        raise FreshR1Error("execution recipe entrypoint does not match the launch request")
    if execution_recipe["entrypoint_source_sha256"] != source_sha256:
        raise FreshR1Error(
            "execution recipe entrypoint source digest does not match the launch request"
        )
    if resource_prediction is None:
        recipe_prediction = execution_recipe.get("resource_prediction")
        if recipe_prediction is not None:
            if not isinstance(recipe_prediction, Mapping):
                raise FreshR1Error("execution recipe resource_prediction must be a mapping")
            resource_prediction = recipe_prediction
    if prefix_contract_path is not None and not resource_telemetry:
        raise FreshR1Error("fixed-batch prefix requires resource telemetry")
    if resource_prediction is not None and not isinstance(resource_prediction, Mapping):
        raise FreshR1Error("resource_prediction must be a mapping")
    validated_gpu_id = _validated_gpu_id(gpu_id)
    validated_cuda_visible_devices = _single_cuda_visible_device(cuda_visible_devices)
    if validated_gpu_id is not None and validated_cuda_visible_devices is not None:
        raise FreshR1Error("gpu_id and cuda_visible_devices are mutually exclusive")
    validated_reservation_evidence = validate_gpu_reservation_evidence(
        gpu_reservation_evidence,
        cuda_visible_devices=validated_cuda_visible_devices,
        physical_gpu_selector=(
            str(validated_gpu_id) if validated_gpu_id is not None else None
        ),
        run_id=run_id,
    )
    if final_worker_ceiling_seconds is None:
        hard_worker_ceiling = MAX_WORKER_CEILING_SECONDS
    else:
        if (
            isinstance(final_worker_ceiling_seconds, bool)
            or not isinstance(final_worker_ceiling_seconds, int)
            or final_worker_ceiling_seconds < 1
        ):
            raise FreshR1Error(
                "final_worker_ceiling_seconds must be a positive integer"
            )
        hard_worker_ceiling = final_worker_ceiling_seconds
    hard_worker_ceiling = min(MAX_WORKER_CEILING_SECONDS, hard_worker_ceiling)
    candidate_deadline_seconds = resolve_candidate_deadline_seconds(
        default_seconds=timeout_seconds,
        prediction=resource_prediction,
        final_worker_ceiling_seconds=hard_worker_ceiling,
    )
    if recbole_commit_identity is not None and recbole_commit_identity != (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    ):
        raise FreshR1Error("delegated RecBole commit identity mismatch")
    run_root = side_root / "experiments" / run_id
    result_root = run_root / "worker"
    checkpoint_dir = result_root / "checkpoints"
    runtime_view = run_root / "runtime_view"
    runtime_view.mkdir(parents=True)
    _symlink_new(repo_root / "scripts", runtime_view / "scripts")
    _symlink_new(repo_root / "configs", runtime_view / "configs")
    if candidate_root is not None:
        _symlink_new(candidate_root / "recclaw_ext", runtime_view / "recclaw_ext")
    else:
        _symlink_new(repo_root / "recclaw_ext", runtime_view / "recclaw_ext")
    capability = build_training_filesystem_capability(
        instance_private_root=side_root,
        result_root=result_root,
        checkpoint_root=checkpoint_dir,
        project_root=runtime_view,
        recbole_root=RECBole_ROOT,
        dataset_root=SEARCH_DATASET_ROOT,
    )
    materialize_training_filesystem_capability(capability)
    capability_path = run_root / "filesystem_capability.v2.json"
    _write_new_json(capability_path, capability.to_dict())
    release_digest = campaign_training_runtime_release().digest
    recbole_identity = recbole_source_identity(RECBole_ROOT)
    if (
        expected_recbole_source_tree_digest is not None
        and recbole_identity["source_tree_digest"]
        != expected_recbole_source_tree_digest
    ):
        raise FreshR1Error("delegated RecBole source tree identity mismatch")
    _write_new_json(run_root / "recbole_source_identity.json", recbole_identity)
    dataset_manifest_digest = bytes_sha256(
        (SEARCH_DATA_ROOT / "search_partition_manifest.json").read_bytes()
    )
    runtime_binding_digest = sha256_digest(
        {
            "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
            "recbole_commit": (
                recbole_commit_identity
                if recbole_commit_identity is not None
                else _git(RECBole_ROOT, "rev-parse", "HEAD")
            ),
            "recbole_source_tree_digest": recbole_identity[
                "source_tree_digest"
            ],
            "runtime_release_digest": release_digest,
            "search_partition": EXPECTED_SEARCH_FILES,
        }
    )
    claim_id = f"{run_identity}-claim:{run_id}"
    round_id = f"{run_identity}-round:{run_id}"
    permit_digest = sha256_digest({"authority": authority})
    prefix_contract_digest = (
        bytes_sha256(prefix_contract_path.read_bytes())
        if prefix_contract_path is not None
        else None
    )
    try:
        experiment_binding = ExperimentBindingV1.from_execution_recipe(
            execution_recipe,
            candidate_root=candidate_root,
            dataset_manifest_digest=dataset_manifest_digest,
            seed=seed,
            epochs=epochs,
            timeout_seconds=candidate_deadline_seconds,
            execution_purpose=execution_purpose,
            resource_telemetry=resource_telemetry,
            watchdog_seconds=watchdog_seconds,
            prefix_contract_digest=prefix_contract_digest,
            run_id=run_id,
            round_id=round_id,
            claim_id=claim_id,
            permit_digest=permit_digest,
            runtime_binding_digest=runtime_binding_digest,
            runtime_release_digest=release_digest,
            runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
            filesystem_capability_digest=capability.capability_digest,
        )
    except ExperimentBindingError as error:
        raise FreshR1Error(str(error)) from error
    binding_path = run_root / "experiment_binding.json"
    _write_new_json(binding_path, experiment_binding.canonical_dict())
    recipe_path = run_root / "execution_recipe.json"
    worker_recipe = _worker_recipe_for_gpu_selection(
        experiment_binding.worker_recipe(),
        validated_gpu_id,
    )
    _write_new_json(recipe_path, worker_recipe)
    start_identity = {
        "binding_digest": experiment_binding.digest,
        "claim_id": experiment_binding.claim_id,
        "execution_purpose": experiment_binding.execution_purpose,
        "ordinary_launch_attempt_ordinal": 1,
        "permit_digest": experiment_binding.permit_digest,
        "round_id": experiment_binding.round_id,
        "run_id": experiment_binding.run_id,
        "runner_abi": experiment_binding.runner_abi,
        "runtime_binding_digest": experiment_binding.runtime_binding_digest,
        "runtime_release_digest": experiment_binding.runtime_release_digest,
    }
    output_path = result_root / "worker_result.json"
    telemetry_path = result_root / "resource_telemetry.json"
    log_path = Path(capability.log_root) / "training.log"
    confirmation_path = result_root / "start_confirmation.json"
    gate_path = result_root / "start_gate.json"
    command = render_campaign_worker_command(
        experiment_binding,
        python_executable=PYTHON_EXECUTABLE,
        worker_path=repo_root / "scripts/campaign_train_worker.py",
        checkpoint_dir=checkpoint_dir,
        data_path=SEARCH_DATA_ROOT,
        execution_recipe_path=recipe_path,
        filesystem_capability_path=capability_path,
        log_path=log_path,
        output_path=output_path,
        project_root=runtime_view,
        recbole_root=RECBole_ROOT,
        start_confirmation_path=confirmation_path,
        start_gate_path=gate_path,
        resource_telemetry_path=(telemetry_path if resource_telemetry else None),
        prefix_contract_path=prefix_contract_path,
    )
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        command,
        cwd=capability.run_working_directory,
        env=_worker_environment(
            capability,
            candidate_root,
            cuda_visible_devices=validated_cuda_visible_devices,
            gpu_id=validated_gpu_id,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def finalize_process_outcome(
        value: Mapping[str, Any],
        *,
        ended_ns: int | None = None,
        device_evidence_consistent: bool = True,
    ) -> dict[str, Any]:
        enriched_value = dict(value)
        enriched_value.setdefault(
            "candidate_deadline_seconds",
            candidate_deadline_seconds,
        )
        enriched_value.setdefault(
            "final_worker_ceiling_seconds",
            hard_worker_ceiling,
        )
        enriched_value.setdefault(
            "resource_prediction",
            (
                canonical_value(dict(resource_prediction))
                if isinstance(resource_prediction, Mapping)
                else None
            ),
        )
        return _finalize_process_outcome(
            enriched_value,
            started_ns=started_ns,
            ended_ns=ended_ns,
            cuda_visible_devices=validated_cuda_visible_devices,
            gpu_id=validated_gpu_id,
            reservation_evidence=validated_reservation_evidence,
            device_evidence_consistent=device_evidence_consistent,
        )

    deadline = time.monotonic() + 30
    while not confirmation_path.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            return finalize_process_outcome({
                "experiment_binding": experiment_binding.canonical_dict(),
                "experiment_binding_digest": experiment_binding.digest,
                "experiment_binding_ref": experiment_binding.ref,
                "execution_recipe_digest": experiment_binding.execution_recipe_digest,
                "recbole_source_identity": recbole_identity,
                "error_message": "worker exited before START_CONFIRMED",
                "exit_status": "RUNTIME_FAILURE",
                "failure_scope": "WORKER_TRANSIENT",
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                ),
                "censoring_trigger": "ENGINEERING_STARTUP",
                "mechanism_effect_update_allowed": False,
                "launcher_return_code": process.returncode,
                "stderr_digest": sha256_digest(stderr),
                "stdout_digest": sha256_digest(stdout),
                "seed": experiment_binding.seed,
            }, ended_ns=time.monotonic_ns())
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            return finalize_process_outcome({
                "experiment_binding": experiment_binding.canonical_dict(),
                "experiment_binding_digest": experiment_binding.digest,
                "experiment_binding_ref": experiment_binding.ref,
                "execution_recipe_digest": experiment_binding.execution_recipe_digest,
                "recbole_source_identity": recbole_identity,
                "error_message": "worker START_CONFIRMED timeout",
                "exit_status": "RUNTIME_FAILURE",
                "failure_scope": "WORKER_TRANSIENT",
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                ),
                "censoring_trigger": "ENGINEERING_STARTUP",
                "mechanism_effect_update_allowed": False,
                "launcher_return_code": 124,
                "seed": experiment_binding.seed,
            }, ended_ns=time.monotonic_ns())
        time.sleep(0.05)
    confirmation = _read_json(confirmation_path)
    if int(confirmation.get("pid", -1)) != process.pid:
        process.kill()
        process.wait()
        return finalize_process_outcome(
            {
                "experiment_binding": experiment_binding.canonical_dict(),
                "experiment_binding_digest": experiment_binding.digest,
                "experiment_binding_ref": experiment_binding.ref,
                "execution_recipe_digest": experiment_binding.execution_recipe_digest,
                "recbole_source_identity": recbole_identity,
                "error_message": "training START_CONFIRMED pid mismatch",
                "exit_status": "RUNTIME_FAILURE",
                "failure_scope": "CANDIDATE_LOCAL",
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                ),
                "censoring_trigger": "ENGINEERING_STARTUP",
                "mechanism_effect_update_allowed": False,
                "launcher_return_code": 124,
                "seed": experiment_binding.seed,
            },
            ended_ns=time.monotonic_ns(),
        )
    _write_start_gate(gate_path, start_identity)
    (
        stdout,
        stderr,
        return_code,
        health_decision,
        durable_telemetry,
        supervisor_terminated,
    ) = (
        _supervise_training_process(
            process,
            telemetry_path=telemetry_path,
            candidate_deadline_seconds=candidate_deadline_seconds,
            final_worker_ceiling_seconds=hard_worker_ceiling,
            prediction=resource_prediction,
            telemetry_enabled=resource_telemetry,
            epochs_requested=epochs,
        )
    )
    process_ended_ns = time.monotonic_ns()
    health_action = health_decision.get("action")
    health_terminated = supervisor_terminated
    censoring_trigger = None
    if health_terminated:
        censoring_trigger = (
            "ENGINEERING_WATCHDOG"
            if health_action == HEALTH_ACTION_HARD_CEILING
            else "ENGINEERING_HEALTH_SUPERVISOR"
        )
    worker = _read_json(output_path) if output_path.is_file() else {
        "error_message": "worker result missing",
        "exit_status": "RUNTIME_FAILURE",
    }
    if durable_telemetry is None and telemetry_path.is_file():
        durable_telemetry = _read_optional_json(telemetry_path)
    metrics, metric_identity_matches = _round_test_feedback_metrics(worker)
    exit_status = "RESOURCE_CENSORED" if health_terminated else worker.get("exit_status")
    if exit_status == "SUCCESS" and not metric_identity_matches:
        exit_status = "RUNTIME_FAILURE"
        return_code = 126
    device_evidence = worker.get("training_device_evidence")
    device_evidence_error = _cross_check_training_device_evidence(
        device_evidence,
        cuda_visible_devices=validated_cuda_visible_devices,
        gpu_id=validated_gpu_id,
        reservation_evidence=validated_reservation_evidence,
    )
    device_evidence_consistent = device_evidence_error is None
    if device_evidence_error is not None:
        exit_status = "RUNTIME_FAILURE"
        return_code = 126
    result = {
        "binding_digest": experiment_binding.digest,
        "candidate_deadline_seconds": candidate_deadline_seconds,
        "experiment_binding": experiment_binding.canonical_dict(),
        "experiment_binding_digest": experiment_binding.digest,
        "experiment_binding_ref": experiment_binding.ref,
        "device_evidence": device_evidence,
        "training_device_evidence": device_evidence,
        "device_evidence_validation": (
            "CONTRADICTORY"
            if device_evidence_error is not None
            else "AVAILABLE_AND_CONSISTENT"
            if isinstance(device_evidence, Mapping)
            else "UNAVAILABLE"
        ),
        "epochs_requested": epochs,
        "execution_recipe_digest": experiment_binding.execution_recipe_digest,
        "exit_status": exit_status,
        "filesystem_mount_audit": worker.get("filesystem_mount_audit"),
        "final_worker_ceiling_seconds": hard_worker_ceiling,
        "launcher_return_code": return_code,
        "log_sha256": bytes_sha256(log_path.read_bytes()) if log_path.is_file() else None,
        "metrics": metrics,
        "metric_source": worker.get("metric_source"),
        "online_partition_role": worker.get("online_partition_role"),
        "result_sha256": bytes_sha256(output_path.read_bytes()) if output_path.is_file() else None,
        "runtime_binding_digest": experiment_binding.runtime_binding_digest,
        "runtime_release_digest": experiment_binding.runtime_release_digest,
        "resource_prediction": (
            canonical_value(dict(resource_prediction))
            if isinstance(resource_prediction, Mapping)
            else None
        ),
        "seed": experiment_binding.seed,
        "stderr_digest": sha256_digest(stderr),
        "stdout_digest": sha256_digest(stdout),
        "training_health": health_decision,
        "failure_scope": classify_failure_scope(
            trigger=health_decision.get("trigger"),
            telemetry=durable_telemetry,
            worker=worker,
        ),
        "worker_error_message": (
            "training device evidence mismatch: " + device_evidence_error
            if device_evidence_error is not None
            else (
                "worker metric identity mismatch"
                if worker.get("exit_status") == "SUCCESS"
                and not metric_identity_matches
                else worker.get("error_message")
            )
        ),
        "worker_error_type": worker.get("error_type"),
    }
    if resource_telemetry:
        result.update(
            {
                "resource_telemetry": (
                    durable_telemetry
                    if durable_telemetry is not None
                    else worker.get("resource_telemetry")
                ),
                "resource_telemetry_sha256": (
                    bytes_sha256(telemetry_path.read_bytes())
                    if telemetry_path.is_file()
                    else None
                ),
            }
        )
    result["recbole_source_identity"] = recbole_identity
    if watchdog_seconds is not None:
        result.update(
            {
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                    if censoring_trigger is not None
                    else None
                ),
                "censoring_trigger": censoring_trigger,
                "failure_scope": classify_failure_scope(
                    trigger=health_decision.get("trigger"),
                    telemetry=durable_telemetry,
                    worker=worker,
                ),
                "mechanism_effect_update_allowed": (
                    False
                    if health_terminated
                    else health_decision.get("mechanism_effect_update_allowed")
                ),
                "resource_deadline_seconds": candidate_deadline_seconds,
                "watchdog_seconds": watchdog_seconds,
                "final_worker_ceiling_seconds": hard_worker_ceiling,
            }
        )
    if health_terminated:
        result.update(
            {
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                ),
                "censoring_trigger": censoring_trigger,
                "failure_scope": classify_failure_scope(
                    trigger=health_decision.get("trigger"),
                    telemetry=durable_telemetry,
                    worker=worker,
                ),
                "mechanism_effect_update_allowed": False,
                "resource_deadline_seconds": candidate_deadline_seconds,
                "watchdog_seconds": watchdog_seconds,
                "final_worker_ceiling_seconds": hard_worker_ceiling,
            }
        )
    if result.get("exit_status") != "SUCCESS":
        result.setdefault("mechanism_effect_update_allowed", False)
    if device_evidence_error is not None:
        result.update(
            {
                "failure_scope": "CANDIDATE_LOCAL",
                "mechanism_effect_update_allowed": False,
                "device_evidence_failure": "CONTRADICTORY_PHYSICAL_OR_LOGICAL_IDENTITY",
            }
        )
    return finalize_process_outcome(
        result,
        ended_ns=process_ended_ns,
        device_evidence_consistent=device_evidence_consistent,
    )


def _episode(
    *,
    side: str,
    slot_id: str,
    spec: Any,
    capability: Any,
    qualification: MechanicalQualificationRun,
    candidate_run: Mapping[str, Any],
    baseline_run: Mapping[str, Any],
) -> TypedResearchEpisodeV1:
    outcome = canonical_value(
        {
            "baseline_metrics": baseline_run["metrics"],
            "candidate_metrics": candidate_run["metrics"],
            "metric": "ndcg@10",
            "partition": "DEVELOPMENT_VALIDATION",
            "seed": candidate_run["seed"],
            "single_seed_interpretation": "INCONCLUSIVE",
        }
    )
    cost = canonical_value(
        {
            "baseline_wall_time_ms": baseline_run["wall_time_ms"],
            "candidate_wall_time_ms": candidate_run["wall_time_ms"],
            "physical_training_runs": 2,
        }
    )
    profile = canonical_value(
        {
            "candidate_package_digest": capability.candidate_package_digest,
            "current_profile_digest": spec.current_profile_digest,
            "mode": "R1_DEVELOPMENT_COMPARISON_ONLY",
        }
    )
    return TypedResearchEpisodeV1(
        campaign_id=f"{CORRECTED_RUN_IDENTITY}-{side}-{slot_id}",
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        hypothesis=spec.hypothesis,
        executable_capability_ref=capability.capability_id,
        executable_capability_digest=capability.digest,
        executable_profile_ref=(
            f"{CORRECTED_RUN_IDENTITY}-development-profile:{sha256_digest(profile)}"
        ),
        executable_profile_digest=sha256_digest(profile),
        experiment_binding_ref=candidate_run["experiment_binding_ref"],
        experiment_binding_digest=candidate_run["experiment_binding_digest"],
        comparator_ref=baseline_run["experiment_binding_ref"],
        comparator_digest=baseline_run["experiment_binding_digest"],
        outcome_ref=(
            f"{CORRECTED_RUN_IDENTITY}-development-outcome:{sha256_digest(outcome)}"
        ),
        outcome_digest=sha256_digest(outcome),
        cost_ref=(
            f"{CORRECTED_RUN_IDENTITY}-development-cost:{sha256_digest(cost)}"
        ),
        cost_digest=sha256_digest(cost),
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        experiment_executed=True,
        mechanism_interpretation="NOT_ADJUDICATED",
        competing_explanation=spec.competing_explanation,
        failure_class=ResearchFailureClassV1.INCONCLUSIVE,
        mechanism_negative_evidence=False,
        next_discriminative_test=spec.falsifier,
        qualification_receipt_ref=qualification.receipt.receipt_id,
        qualification_receipt_digest=qualification.receipt.digest,
        qualification_evidence_used_as_scientific=False,
    )


def _shared_policy(
    implementation_prompt_digest: str,
    tool_policy_digest: str,
    *,
    allowed_files: tuple[str, ...] | None = None,
    execution_contract: Mapping[str, Any] | None = None,
) -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=allowed_files
        or ("recclaw_ext/__init__.py", "recclaw_ext/candidate.py"),
        dependency_identity_ref=(
            "repo:docs/research_line/vnext/R1_R2_RUNTIME_DEPENDENCY_LOCK_V1#dependency_identity"
        ),
        dependency_identity_digest=(
            "f007435e68fc3f7f45baa8f8bb9ca093883545d365c1a0e6684db79ba69fb5a3"
        ),
        runtime_identity_ref=(
            "repo:docs/research_line/vnext/R1_R2_RUNTIME_DEPENDENCY_LOCK_V1#runtime_identity"
        ),
        runtime_identity_digest=(
            "386429298191030628920a8312da8bd914659ae6e72d5a99a2138c4be176d09e"
        ),
        prompt_digest=implementation_prompt_digest,
        tool_policy_digest=tool_policy_digest,
        implementation_token_ceiling=IMPLEMENTATION_TOKEN_CEILING,
        execution_contract=execution_contract,
    )


def _failure_record(
    *,
    side: str,
    slot_id: str,
    stage: str,
    failure_class: str,
    reason_code: str,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return canonical_value(
        {
            "detail": detail,
            "failure_class": failure_class,
            "mechanism_negative_evidence": False,
            "reason_code": reason_code,
            "schema": "recclaw.research-line.fresh-r1-diagnostic-failure.v1",
            "side": side,
            "slot_id": slot_id,
            "stage": stage,
        }
    )


def _negative_fixture_check(provider_schema: Mapping[str, Any]) -> dict[str, Any]:
    fixture_path = _resource_root() / "fresh_open_spec_v4_duplicate_arrays_negative_fixture.json"
    fixture = _read_json(fixture_path)
    proposal = fixture["proposals"][0]
    proposal["compatibility_requirements"] = [
        PROTOCOL_REQUIREMENTS[0],
        PROTOCOL_REQUIREMENTS[0],
    ]
    proposal["resolution_facts"]["required_dependencies"] = [
        AVAILABLE_DEPENDENCIES[0],
        AVAILABLE_DEPENDENCIES[0],
    ]
    try:
        validate_v4_response_contract(fixture, provider_schema=provider_schema)
    except V4LocalUniquenessError as error:
        return {
            "fixture_sha256": bytes_sha256(fixture_path.read_bytes()),
            "normalized_fixture_digest": sha256_digest(fixture),
            "observed_error": type(error).__name__,
            "status": "PASS_EXPECTED_REJECTION",
        }
    raise FreshR1Error("required duplicate-array negative fixture did not fail")


def _proposal_record_base(side: str, slot: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "denominator_included": True,
        "logical_slot_id": slot["logical_slot_id"],
        "producer_role": slot["producer_role"],
        "proposal_seed": seed,
        "side": side,
    }


def _physical_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    attempts = [attempt for row in records for attempt in row.get("provider_attempts", ())]
    return {
        "billed_tokens": sum(int(row.get("billed_tokens") or 0) for row in attempts),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in attempts),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in attempts),
        "physical_calls": sum(int(row.get("physical_call_count", 1)) for row in attempts),
        "provider_wall_time_ms": sum(int(row.get("latency_ms") or 0) for row in attempts),
        "retries": sum(max(0, len(row.get("provider_attempts", ())) - 1) for row in records),
    }


def run_formal_fresh_r1(repo_root: Path, *, canonical_receipt_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    started_ns = time.monotonic_ns()
    identity = verify_formal_identity(repo_root)
    resource_root = _resource_root()
    docs_root = repo_root / "docs/research_line/vnext"
    manifest = _read_json(docs_root / "R1_R2_PREFREEZE_MANIFEST_V11.json")
    identity_plan = _read_json(docs_root / "R1_R2_FRESH_IDENTITY_PLAN_V1.json")
    schedule = _read_json(resource_root / "fresh_open_spec_call_schedule_v1.json")
    proposal_template_path = resource_root / "fresh_open_spec_proposal_prompt_v1.txt"
    proposal_template = proposal_template_path.read_text(encoding="utf-8")
    base_proposal_schema_path = (
        resource_root / "fresh_open_spec_proposal_response_v4_provider.schema.json"
    )
    proposal_schema_delta_path = (
        resource_root / "fresh_r1_proposal_schema_delta_v1.json"
    )
    implementation_template_path = resource_root / "fresh_r1_implementer_prompt_v1.txt"
    implementation_template = implementation_template_path.read_text(encoding="utf-8")
    implementation_schema_path = resource_root / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resource_root / "fresh_open_spec_tool_policy_v1.json"
    implementation_prompt_digest = bytes_sha256(implementation_template_path.read_bytes())
    tool_policy_digest = bytes_sha256(tool_policy_path.read_bytes())
    policy = _shared_policy(implementation_prompt_digest, tool_policy_digest)
    jsonschema.validators.validator_for(_read_json(implementation_schema_path)).check_schema(
        _read_json(implementation_schema_path)
    )
    if bytes_sha256(proposal_template_path.read_bytes()) != manifest["exact_provider_contract"]["prompt_digest"]:
        raise FreshR1Error("proposal prompt template digest differs from V11")
    if bytes_sha256(base_proposal_schema_path.read_bytes()) != manifest["exact_provider_contract"]["response_schema_digest"]:
        raise FreshR1Error("proposal schema digest differs from V11")
    if bytes_sha256(tool_policy_path.read_bytes()) != manifest["exact_provider_contract"]["tool_policy_digest"]:
        raise FreshR1Error("proposal tool policy digest differs from V11")
    proposal_schema_delta = _read_json(proposal_schema_delta_path)
    if proposal_schema_delta.get("base_schema_sha256") != bytes_sha256(
        base_proposal_schema_path.read_bytes()
    ):
        raise FreshR1Error("corrected R1 schema delta targets a different base")
    proposal_schema = derive_fresh_r1_proposal_schema(
        _read_json(base_proposal_schema_path),
        proposal_schema_delta,
    )
    negative_fixture = _negative_fixture_check(proposal_schema)
    R1_ROOT.mkdir(parents=True)
    _write_new_json(R1_ROOT / "RUN_IDENTITY.json", identity)
    proposal_schema_path = R1_ROOT / "contracts/fresh_r1_proposal_response.schema.json"
    proposal_schema_digest = _write_new_json(proposal_schema_path, proposal_schema)
    bindings = frozen_search_bindings(
        context_ref=CONTEXT_REF,
        context_digest=CONTEXT_DIGEST,
    )
    environment = frozen_search_resolver_environment(
        available_dependencies=AVAILABLE_DEPENDENCIES,
        budget_limits=BUDGET_LIMITS,
        protocol_requirements=PROTOCOL_REQUIREMENTS,
    )
    seed_plan = identity_plan["r1"]["seed_plan"]
    side_specs = {
        "side_a": {
            "identity": identity_plan["r1"]["side_a"]["identity"],
            "proposal_seeds": seed_plan["side_a_proposal_seed_by_slot"],
            "qualification_seeds": seed_plan["qualification_seed_by_slot_side_a"],
        },
        "side_b": {
            "identity": identity_plan["r1"]["side_b"]["identity"],
            "proposal_seeds": seed_plan["side_b_proposal_seed_by_slot"],
            "qualification_seeds": seed_plan["qualification_seed_by_slot_side_b"],
        },
    }
    records: dict[str, list[dict[str, Any]]] = {"side_a": [], "side_b": []}
    live_specs: dict[tuple[str, str], tuple[Any, Mapping[str, Any], Any]] = {}
    seen_spec_digests: set[str] = set()
    proposal_contracts = {
        side: call_contract_for_side(
            side,
            service="proposal",
            prompt_digest=bytes_sha256(proposal_template_path.read_bytes()),
            response_schema_digest=proposal_schema_digest,
        )
        for side in ("side_a", "side_b")
    }
    implementation_schema_digest = bytes_sha256(
        implementation_schema_path.read_bytes()
    )
    implementation_contracts = {
        side: call_contract_for_side(
            side,
            service="implementation",
            prompt_digest=implementation_prompt_digest,
            response_schema_digest=implementation_schema_digest,
        )
        for side in ("side_a", "side_b")
    }
    if proposal_contracts["side_a"] != proposal_contracts["side_b"]:
        raise FreshR1Error("proposal A/B call contracts differ")
    if implementation_contracts["side_a"] != implementation_contracts["side_b"]:
        raise FreshR1Error("implementation A/B call contracts differ")

    # Proposal generation is completed before any implementation or qualification
    # outcome exists, keeping all sixteen frozen slots outcome-independent.
    for slot_index, slot in enumerate(schedule["role_schedule"]):
        for side in ("side_a", "side_b"):
            side_config = side_specs[side]
            slot_id = str(slot["logical_slot_id"])
            seed = int(side_config["proposal_seeds"][slot_index])
            record = _proposal_record_base(side, slot, seed)
            prompt = render_proposal_prompt(
                proposal_template,
                side_identity=str(side_config["identity"]),
                logical_slot_id=slot_id,
                proposal_seed=seed,
                producer_role=str(slot["producer_role"]),
            )
            logical_id = f"{CORRECTED_RUN_IDENTITY}:{side}:{slot_id}:proposal"
            call_result = bounded_provider_call(
                call_root=R1_ROOT / side / "provider/proposals" / slot_id,
                schema_path=proposal_schema_path,
                logical_call_id=logical_id,
                session_id=f"{CORRECTED_RUN_IDENTITY}:{side}:proposal-session",
                prompt=prompt,
                token_ceiling=PROPOSAL_TOKEN_CEILING,
            )
            record["provider_attempts"] = call_result.attempts
            if call_result.call is None:
                failure = _failure_record(
                    side=side,
                    slot_id=slot_id,
                    stage="PROPOSAL_PROVIDER",
                    failure_class="PROVIDER",
                    reason_code="PROVIDER_CALL_FAILED",
                    detail=call_result.failure,
                )
                record.update(
                    {
                        "missingness_class": "MISSING_PROVIDER_OR_ENGINEERING_FAILURE",
                        "slot_status": "MISSING_PROVIDER_FAILURE",
                        "failure": failure,
                    }
                )
                _write_new_json(R1_ROOT / side / "slots" / f"{slot_id}.json", record)
                records[side].append(record)
                continue
            call = call_result.call
            record["proposal_response_digest"] = call.response_digest
            record["returned_model"] = call.returned_model
            try:
                validate_v4_response_contract(call.response, provider_schema=proposal_schema)
                draft = call.response["proposals"][0]
                if draft["producer_role"] != slot["producer_role"]:
                    raise FreshR1Error("Provider changed the preassigned Producer role")
                spec, facts = project_open_producer_draft(draft, bindings=bindings)
                resolution = resolve_capability(
                    spec,
                    resolution_facts=facts,
                    environment=environment,
                )
                if spec.digest in seen_spec_digests:
                    raise FreshR1Error("Provider returned a duplicate fresh spec")
                seen_spec_digests.add(spec.digest)
                record.update(
                    {
                        "resolution": resolution.resolution.value,
                        "resolution_digest": resolution.digest,
                        "resolution_reason_codes": resolution.reason_codes,
                        "spec_digest": spec.digest,
                        "spec_ref": spec.spec_id,
                        "slot_status": (
                            "SPEC_READY_FOR_IMPLEMENTATION"
                            if resolution.resolution is CapabilityResolutionResultV1.INNOVATION_REQUIRED
                            else "TERMINAL_RESOLUTION"
                        ),
                    }
                )
                _write_new_json(
                    R1_ROOT / side / "specs" / f"{slot_id}.json",
                    {
                        "resolution": resolution.canonical_dict(),
                        "research_spec": spec.canonical_dict(),
                        "resolution_facts": facts,
                    },
                )
                if resolution.resolution is CapabilityResolutionResultV1.INNOVATION_REQUIRED:
                    live_specs[(side, slot_id)] = (spec, facts, resolution)
            except (FreshR1Error, V4LocalUniquenessError, jsonschema.ValidationError, ValueError) as error:
                record.update(
                    {
                        "failure": _failure_record(
                            side=side,
                            slot_id=slot_id,
                            stage="PROPOSAL_SEMANTIC_CONTRACT",
                            failure_class="PROVIDER",
                            reason_code=type(error).__name__.upper(),
                            detail={"detail_digest": sha256_digest(str(error))},
                        ),
                        "missingness_class": "MISSING_PROVIDER_OR_ENGINEERING_FAILURE",
                        "slot_status": "TERMINAL_RESPONSE_CONTRACT_FAILURE",
                    }
                )
            _write_new_json(R1_ROOT / side / "slots" / f"{slot_id}.json", record)
            records[side].append(record)

    # Shared blind implementation and qualification consume every valid spec in
    # the original side/slot order. No result is used to select another slot.
    for side in ("side_a", "side_b"):
        side_root = R1_ROOT / side
        for slot_index, slot in enumerate(schedule["role_schedule"]):
            slot_id = str(slot["logical_slot_id"])
            key = (side, slot_id)
            if key not in live_specs:
                continue
            record = records[side][slot_index]
            spec, facts, resolution = live_specs[key]
            request = build_shared_implementer_request(spec, policy=policy)
            prompt = render_implementation_prompt(implementation_template, request)
            call_result = bounded_provider_call(
                call_root=side_root / "provider/implementations" / slot_id,
                schema_path=implementation_schema_path,
                logical_call_id=(
                    f"{CORRECTED_RUN_IDENTITY}:{side}:{slot_id}:implementation"
                ),
                session_id=(
                    f"{CORRECTED_RUN_IDENTITY}:"
                    "shared-origin-blind-implementation-session"
                ),
                prompt=prompt,
                token_ceiling=IMPLEMENTATION_TOKEN_CEILING,
            )
            record["implementation_provider_attempts"] = call_result.attempts
            if call_result.call is None:
                record.update(
                    {
                        "failure": _failure_record(
                            side=side,
                            slot_id=slot_id,
                            stage="IMPLEMENTATION_PROVIDER",
                            failure_class="PROVIDER",
                            reason_code="PROVIDER_CALL_FAILED",
                            detail=call_result.failure,
                        ),
                        "slot_status": "IMPLEMENTATION_PROVIDER_FAILURE",
                    }
                )
                continue
            implementation_response = call_result.call.response
            implementation = implementation_response["proposals"][0]
            try:
                materialized, qualification, behavior = _materialize_and_qualify(
                    repo_root=repo_root,
                    side_root=side_root,
                    slot_id=slot_id,
                    seed=int(side_specs[side]["qualification_seeds"][slot_index]),
                    spec=spec,
                    implementation=implementation,
                    implementation_prompt_digest=implementation_prompt_digest,
                    tool_policy_digest=tool_policy_digest,
                )
            except InnovationSpineError as error:
                record.update(
                    {
                        "failure": _failure_record(
                            side=side,
                            slot_id=slot_id,
                            stage="IMPLEMENTATION_MATERIALIZATION",
                            failure_class=error.failure_class,
                            reason_code=error.reason_code,
                            detail={"detail_digest": sha256_digest(str(error))},
                        ),
                        "slot_status": "IMPLEMENTATION_FAILURE",
                    }
                )
                continue
            qualification_payload = qualification.to_dict()
            _write_new_json(
                side_root / "qualifications" / f"{slot_id}.json",
                {
                    **qualification_payload,
                    "behavioral_mechanism_evidence": behavior,
                },
            )
            record.update(
                {
                    "candidate_package_digest": materialized.package.digest,
                    "candidate_package_ref": materialized.package.package_id,
                    "qualification_receipt_digest": qualification.receipt.digest,
                    "qualification_status": qualification.receipt.status.value,
                    "qualification_stage": qualification.receipt.stage.value,
                    "behavioral_mechanism_evidence": behavior,
                    "slot_status": (
                        "QUALIFIED"
                        if qualification.receipt.status is QualificationStatusV1.PASS
                        else "QUALIFICATION_FAILURE"
                    ),
                }
            )
            if qualification.receipt.status is not QualificationStatusV1.PASS:
                record["failure"] = _failure_record(
                    side=side,
                    slot_id=slot_id,
                    stage="QUALIFICATION",
                    failure_class=qualification.receipt.failure_class.value,
                    reason_code=str(
                        (qualification.failure_detail or {}).get(
                            "reason_code", "QUALIFICATION_FAILED"
                        )
                    ),
                    detail=qualification.failure_detail,
                )
                continue
            capability = admit_qualified_capability(
                spec,
                materialized.package,
                qualification.receipt,
                capability_kind=_mechanism_kind(facts["high_change_dimensions"]),
                capability_version="fresh-r1-corrected-v1",
                semantic_identity_ref=(
                    f"{CORRECTED_RUN_IDENTITY}-semantic:" + sha256_digest(
                        {
                            "capability_diff": facts["capability_diff"],
                            "high_change_dimensions": facts["high_change_dimensions"],
                        }
                    )
                ),
                semantic_identity_digest=sha256_digest(
                    {
                        "capability_diff": facts["capability_diff"],
                        "high_change_dimensions": facts["high_change_dimensions"],
                    }
                ),
            )
            accepted_dimensions = {
                "MODEL_STRUCTURE",
                "INTERACTION_STRUCTURE",
                "PROPAGATION_MECHANISM",
                "CORE_REPRESENTATION",
                "CORE_RELATION",
            }
            real_change = bool(
                accepted_dimensions.intersection(facts["high_change_dimensions"])
                and behavior.get("probe_status") == "PASS"
                and behavior.get("extra_parameter_names")
            )
            record.update(
                {
                    "capability_digest": capability.digest,
                    "capability_ref": capability.capability_id,
                    "high_change_dimensions": facts["high_change_dimensions"],
                    "real_mechanism_change": real_change,
                }
            )
            live_specs[key] = (spec, materialized, qualification, capability, facts)
            _write_new_json(
                side_root / "capabilities" / f"{slot_id}.json",
                capability.canonical_dict(),
            )

    preliminary_gate = evaluate_gate(records)
    episodes_by_side: dict[str, list[dict[str, Any]]] = {"side_a": [], "side_b": []}
    training_runs: list[dict[str, Any]] = []
    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    baseline_source_sha256 = bytes_sha256(baseline_source.read_bytes())
    for side in ("side_a", "side_b"):
        side_root = R1_ROOT / side
        for slot_index, slot in enumerate(schedule["role_schedule"]):
            slot_id = str(slot["logical_slot_id"])
            key = (side, slot_id)
            value = live_specs.get(key)
            if value is None or len(value) != 5:
                continue
            spec, materialized, qualification, capability, _facts = value
            seed = int(side_specs[side]["qualification_seeds"][slot_index])
            baseline_run = run_development_training(
                repo_root=repo_root,
                side_root=side_root,
                run_id=f"{slot_id}-matched-bpr",
                seed=seed,
                candidate_root=None,
                entrypoint="recbole.model.general_recommender.bpr:BPR",
                source_sha256=baseline_source_sha256,
                execution_recipe=_bpr_comparator_execution_recipe(
                    spec=spec,
                    run_id=f"{slot_id}-matched-bpr",
                    entrypoint="recbole.model.general_recommender.bpr:BPR",
                    source_sha256=baseline_source_sha256,
                ),
            )
            blind_candidate_id = str(
                materialized.shared_request["blind_candidate_id"]
            )
            candidate_root = (
                side_root / "candidates" / slot_id / blind_candidate_id
            )
            candidate_source = candidate_root / "recclaw_ext/candidate.py"
            candidate_entrypoint = materialized.package.executable_entrypoint
            candidate_source_sha256 = bytes_sha256(candidate_source.read_bytes())
            candidate_run = run_development_training(
                repo_root=repo_root,
                side_root=side_root,
                run_id=f"{slot_id}-candidate",
                seed=seed,
                candidate_root=candidate_root,
                entrypoint=candidate_entrypoint,
                source_sha256=candidate_source_sha256,
                execution_recipe=_fresh_r1_candidate_execution_recipe(
                    spec=spec,
                    capability=capability,
                    package=materialized.package,
                    run_id=f"{slot_id}-candidate",
                    entrypoint=candidate_entrypoint,
                    source_sha256=candidate_source_sha256,
                ),
            )
            training_runs.extend(
                [
                    {"run_kind": "MATCHED_CONTROL", "side": side, "slot_id": slot_id, **baseline_run},
                    {"run_kind": "CANDIDATE", "side": side, "slot_id": slot_id, **candidate_run},
                ]
            )
            if (
                baseline_run.get("exit_status") == "SUCCESS"
                and candidate_run.get("exit_status") == "SUCCESS"
                and "ndcg@10" in baseline_run.get("metrics", {})
                and "ndcg@10" in candidate_run.get("metrics", {})
            ):
                episode = _episode(
                    side=side,
                    slot_id=slot_id,
                    spec=spec,
                    capability=capability,
                    qualification=qualification,
                    candidate_run=candidate_run,
                    baseline_run=baseline_run,
                )
                episode_payload = episode.canonical_dict()
                _write_new_json(side_root / "episodes" / f"{slot_id}.json", episode_payload)
                episodes_by_side[side].append(
                    {"episode_digest": episode.digest, "episode_ref": episode.episode_id}
                )
                records[side][slot_index]["episode_digest"] = episode.digest
                records[side][slot_index]["episode_failure_class"] = episode.failure_class.value
            else:
                records[side][slot_index]["experiment_failure"] = _failure_record(
                    side=side,
                    slot_id=slot_id,
                    stage="DEVELOPMENT_COMPARISON",
                    failure_class="RUNTIME",
                    reason_code="MATCHED_COMPARISON_DID_NOT_CLOSE",
                    detail={
                        "baseline_exit_status": baseline_run.get("exit_status"),
                        "candidate_exit_status": candidate_run.get("exit_status"),
                    },
                )

    final_gate = evaluate_gate(records)
    resolver_distribution = Counter(
        str(row.get("resolution"))
        for side_rows in records.values()
        for row in side_rows
        if row.get("resolution")
    )
    missingness = {
        side: Counter(
            str(row.get("slot_status"))
            for row in side_rows
            if row.get("slot_status") != "QUALIFIED"
        )
        for side, side_rows in records.items()
    }
    episode_counts = {side: len(values) for side, values in episodes_by_side.items()}
    end_to_end = all(count >= 1 for count in episode_counts.values())
    proposal_usage = _physical_usage(
        [row for side_rows in records.values() for row in side_rows]
    )
    implementation_usage = _physical_usage(
        [
            {"provider_attempts": row.get("implementation_provider_attempts", ())}
            for side_rows in records.values()
            for row in side_rows
        ]
    )
    status = "R1_PASS" if final_gate["pass"] and end_to_end else "R1_FAIL"
    receipt = canonical_value(
        {
            "a_b_interpretation": "SYMMETRIC_INDEPENDENT_REPLICATION_LANES_NOT_A_TREATMENT_EFFECT_TEST",
            "attempt_identity": identity,
            "denominator_per_side": 8,
            "end_to_end_result_chain_pass": end_to_end,
            "episode_counts": episode_counts,
            "failure_classification_policy": {
                "engineering_provider_runtime_package_interface_resource_protocol": "DIAGNOSTIC_NOT_MECHANISM_NEGATIVE_EVIDENCE",
                "qualification_evidence_class": "DEVELOPMENT_ONLY",
                "typed_episode_evidence_class": "INCONCLUSIVE_EXPERIMENT",
            },
            "gate": final_gate,
            "held_out_reads": 0,
            "implementation_call_contract_by_side": implementation_contracts,
            "implementation_provider_usage": implementation_usage,
            "manual_candidate_patches": 0,
            "missingness_by_side": {
                side: dict(sorted(values.items())) for side, values in missingness.items()
            },
            "negative_fixture": negative_fixture,
            "preliminary_gate_before_experiments": preliminary_gate,
            "proposal_base_call_contract_digest": manifest["future_r1_scientific_contract"][
                "shared_call_contract_digest"
            ],
            "proposal_call_contract_by_side": proposal_contracts,
            "proposal_schema_delta": {
                "base_schema_sha256": bytes_sha256(
                    base_proposal_schema_path.read_bytes()
                ),
                "delta_sha256": bytes_sha256(
                    proposal_schema_delta_path.read_bytes()
                ),
                "effective_schema_sha256": proposal_schema_digest,
                "purpose": "STRICT_EXACT_TOKEN_PROVIDER_INTERFACE_CORRECTION",
            },
            "proposal_provider_usage": proposal_usage,
            "proposal_slots_per_side": 8,
            "resolver_distribution": dict(sorted(resolver_distribution.items())),
            "retry_policy": {
                "backoff_ms": list(BACKOFF_MS),
                "maximum_physical_attempts": MAX_PHYSICAL_ATTEMPTS,
                "policy_digest": manifest["bounded_retry"]["policy_digest"],
            },
            "prior_sealed_r1_failure_interpretation": (
                "GENERIC_ORCHESTRATION_AND_IMPLEMENTER_INTERFACE_FAILURES_"
                "WITH_NO_MECHANISM_OUTCOME"
            ),
            "schema": (
                "recclaw.research-line.fresh-r1-training-filesystem-fix-v3-"
                "canonical-receipt.v1"
            ),
            "side_records": records,
            "status": status,
            "training": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "epochs_requested_per_run": EXPERIMENT_EPOCHS,
                "held_out_exposed": False,
                "physical_runs": len(training_runs),
                "runs": training_runs,
                "total_wall_time_ms": sum(int(row["wall_time_ms"]) for row in training_runs),
            },
            "typed_research_episode_digests": {
                side: [row["episode_digest"] for row in values]
                for side, values in episodes_by_side.items()
            },
            "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
        }
    )
    external_receipt_sha256 = _write_new_json(R1_ROOT / "R1_CANONICAL_RECEIPT.json", receipt)
    if canonical_receipt_path.exists():
        raise FreshR1Error(f"canonical repository receipt already exists: {canonical_receipt_path}")
    repository_receipt = {
        **receipt,
        "external_receipt_ref": str(R1_ROOT / "R1_CANONICAL_RECEIPT.json"),
        "external_receipt_sha256": external_receipt_sha256,
    }
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "GPU_RESERVATION_EVIDENCE_SCHEMA",
    "GPU_RESERVATION_STATUS_CONTRADICTORY",
    "GPU_RESERVATION_STATUS_MEASURED",
    "GPU_RESERVATION_STATUS_UNMEASURED",
    "GPU_WORKER_SECONDS_SEMANTICS",
    "assess_training_health",
    "classify_failure_scope",
    "FreshR1Error",
    "MAX_WORKER_CEILING_SECONDS",
    "ProviderAttemptResult",
    "bounded_provider_call",
    "evaluate_gate",
    "render_implementation_prompt",
    "render_proposal_prompt",
    "resolve_candidate_deadline_seconds",
    "retry_eligible",
    "run_formal_fresh_r1",
    "validate_gpu_reservation_evidence",
    "verify_formal_identity",
]
