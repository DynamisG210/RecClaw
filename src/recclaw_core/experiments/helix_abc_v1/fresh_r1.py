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
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema

from .canary_broker import CanaryBrokerError, CanaryBrokerCallV1
from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from .capability_admission import admit_qualified_capability
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
from .lab_api_broker import LabApiCanaryBrokerV1, load_lab_api_credentials
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
    http_status = None
    exception_type = None
    receipt: dict[str, Any] = {}
    db_path = private_root / "broker.sqlite3"
    if db_path.is_file():
        connection = sqlite3.connect(db_path)
        try:
            row = connection.execute(
                "SELECT error_detail_json, receipt_json FROM calls"
            ).fetchone()
        finally:
            connection.close()
        if row is not None:
            detail = json.loads(str(row[0])) if row[0] else {}
            receipt = json.loads(str(row[1])) if row[1] else {}
            exception_type = detail.get("exception_type")
            http_status = receipt.get("http_status")
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
            "request_digest": receipt.get("request_digest"),
            "receipt_digest": (
                error.receipt.receipt_digest if error.receipt is not None else None
            ),
        }
    )


def retry_eligible(failure: Mapping[str, Any]) -> bool:
    status = failure.get("http_status")
    if isinstance(status, int) and (status in {408, 429} or status >= 500):
        return True
    if failure.get("failure_class") == "TIMEOUT":
        return True
    return failure.get("exception_type") in {
        "ConnectionResetError",
        "RemoteDisconnected",
    }


def bounded_provider_call(
    *,
    call_root: Path,
    schema_path: Path,
    logical_call_id: str,
    session_id: str,
    prompt: str,
    token_ceiling: int,
    expected_transport_release_digest: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ProviderAttemptResult:
    attempts: list[dict[str, Any]] = []
    request_digests: set[str] = set()
    for ordinal in range(1, MAX_PHYSICAL_ATTEMPTS + 1):
        if ordinal > 1:
            sleep(BACKOFF_MS[ordinal - 2] / 1000)
        private_root = call_root / f"physical_attempt_{ordinal:02d}"
        broker: LabApiCanaryBrokerV1 | None = None
        started = time.monotonic_ns()
        try:
            broker = LabApiCanaryBrokerV1(
                private_root,
                schema_path=schema_path,
                config_path=API_CONFIG,
                model=MODEL,
                max_total_tokens_per_call=token_ceiling,
                timeout_ms=900_000,
                release_manifest_path=None,
            )
            if (
                expected_transport_release_digest is not None
                and broker.release.release_digest != expected_transport_release_digest
            ):
                raise FreshR1Error("proposal transport release identity mismatch")
            call = broker.call_with_session(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=session_id,
                prompt=prompt,
                expected_proposal_count=1,
                max_total_tokens=token_ceiling,
            )
            request_digests.add(call.request_digest)
            if len(request_digests) != 1:
                raise FreshR1Error("bounded retry changed the request payload digest")
            attempts.append(
                {
                    "billed_tokens": call.total_tokens,
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
                    "latency_ms": max(1, (time.monotonic_ns() - started) // 1_000_000),
                    "ordinal": ordinal,
                    "status": "FAILED",
                }
            )
            if not retry_eligible(failure) or ordinal == MAX_PHYSICAL_ATTEMPTS:
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
) -> Callable[[Any, Any, Any], None]:
    def check(model: Any, config: Any, dataset: Any) -> None:
        import torch

        from recbole.data.interaction import Interaction
        from recbole.model.general_recommender.bpr import BPR
        from recbole.utils import InputType, ModelType

        if config["MODEL_TYPE"] is not ModelType.GENERAL:
            raise AssertionError("candidate is not a general recommender")
        if model.input_type is not InputType.PAIRWISE:
            raise AssertionError("candidate is not pairwise")
        overridden = tuple(
            name
            for name in ("calculate_loss", "predict", "full_sort_predict")
            if name in model.__class__.__dict__
        )
        if len(overridden) != 3:
            raise AssertionError("candidate must implement all three behavioral methods")
        baseline = BPR(config, dataset).to(config["device"])
        with torch.no_grad():
            if hasattr(model, "user_embedding") and hasattr(baseline, "user_embedding"):
                baseline.user_embedding.weight.copy_(model.user_embedding.weight)
            if hasattr(model, "item_embedding") and hasattr(baseline, "item_embedding"):
                baseline.item_embedding.weight.copy_(model.item_embedding.weight)
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


def _qualification_fixture(repo_root: Path, *, seed: int, root: Path) -> RecBoleQualificationFixture:
    mini_data = (
        repo_root
        / "tests/experiments/helix_abc_v1/fixtures/innovation_spine/data"
    )
    return RecBoleQualificationFixture(
        project_root=repo_root,
        recbole_root=RECBole_ROOT,
        data_path=mini_data,
        dataset="mini",
        base_model_config="BPR",
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
    unit_check_factory: Callable[
        [dict[str, Any]], Callable[[Any, Any, Any], None]
    ] = _shared_behavioral_unit_check,
) -> tuple[MaterializedCandidate, MechanicalQualificationRun, dict[str, Any]]:
    policy = _shared_policy(implementation_prompt_digest, tool_policy_digest)
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


def _symlink_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), target, target_is_directory=source.is_dir())


def _write_start_gate(path: Path, identity: Mapping[str, Any]) -> None:
    _write_new_json(path, {**identity, "gate_status": "TRAINING_AUTHORIZED"})


def _worker_environment(capability: Any, candidate_root: Path | None) -> dict[str, str]:
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "TZ",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(capability.environment)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    roots = []
    if candidate_root is not None:
        roots.append(candidate_root.as_posix())
    roots.extend([str(Path(__file__).resolve().parents[4] / "src"), str(Path(__file__).resolve().parents[4])])
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
    epochs: int = EXPERIMENT_EPOCHS,
    execution_purpose: str = "DEVELOPMENT_PILOT_OFFLINE_TOPN",
    resource_telemetry: bool = False,
    watchdog_seconds: int | None = None,
    prefix_contract_path: Path | None = None,
) -> dict[str, Any]:
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
    recipe = {
        "base_model_config": "BPR",
        "config": {},
        "entrypoint": entrypoint,
        "entrypoint_source_sha256": source_sha256,
        "mechanism_id": run_id,
        "model": "BPR",
    }
    recipe_path = run_root / "execution_recipe.json"
    _write_new_json(recipe_path, {"execution_recipe": recipe})
    release_digest = campaign_training_runtime_release().digest
    binding = {
        "candidate_source_digest": source_sha256,
        "dataset_manifest_digest": bytes_sha256(
            (SEARCH_DATA_ROOT / "search_partition_manifest.json").read_bytes()
        ),
        "entrypoint": entrypoint,
        "run_id": run_id,
        "seed": seed,
    }
    if (
        epochs != EXPERIMENT_EPOCHS
        or execution_purpose != "DEVELOPMENT_PILOT_OFFLINE_TOPN"
        or resource_telemetry
    ):
        binding.update(
            {
                "epochs": epochs,
                "execution_purpose": execution_purpose,
                "resource_telemetry": resource_telemetry,
            }
        )
    if prefix_contract_path is not None:
        if not resource_telemetry:
            raise FreshR1Error("fixed-batch prefix requires resource telemetry")
        binding["prefix_contract_file_sha256"] = bytes_sha256(
            prefix_contract_path.read_bytes()
        )
    binding_digest = sha256_digest(binding)
    runtime_binding_digest = sha256_digest(
        {
            "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
            "recbole_commit": (
                recbole_commit_identity
                if recbole_commit_identity is not None
                else _git(RECBole_ROOT, "rev-parse", "HEAD")
            ),
            "runtime_release_digest": release_digest,
            "search_partition": EXPECTED_SEARCH_FILES,
        }
    )
    start_identity = {
        "binding_digest": binding_digest,
        "claim_id": f"{run_identity}-claim:{run_id}",
        "execution_purpose": execution_purpose,
        "ordinary_launch_attempt_ordinal": 1,
        "permit_digest": sha256_digest(
            {"authority": authority}
        ),
        "round_id": f"{run_identity}-round:{run_id}",
        "run_id": run_id,
        "runner_abi": CAMPAIGN_TRAINING_RUNNER_ABI,
        "runtime_binding_digest": runtime_binding_digest,
        "runtime_release_digest": release_digest,
    }
    output_path = result_root / "worker_result.json"
    telemetry_path = result_root / "resource_telemetry.json"
    log_path = Path(capability.log_root) / "training.log"
    confirmation_path = result_root / "start_confirmation.json"
    gate_path = result_root / "start_gate.json"
    command = [
        str(PYTHON_EXECUTABLE),
        str(repo_root / "scripts/campaign_train_worker.py"),
        "--binding-digest", binding_digest,
        "--claim-id", str(start_identity["claim_id"]),
        "--checkpoint-dir", str(checkpoint_dir),
        "--data-path", str(SEARCH_DATA_ROOT),
        "--dataset", "ml-1m",
        "--epochs", str(epochs),
        "--execution-purpose", str(start_identity["execution_purpose"]),
        "--execution-recipe-path", str(recipe_path),
        "--filesystem-capability-path", str(capability_path),
        "--filesystem-mode", "HASH_AUDITED_PRIVATE_ROOT_V1",
        "--log-path", str(log_path),
        "--model", "BPR",
        "--output-path", str(output_path),
        "--permit-digest", str(start_identity["permit_digest"]),
        "--project-root", str(runtime_view),
        "--recbole-root", str(RECBole_ROOT),
        "--round-id", str(start_identity["round_id"]),
        "--run-id", run_id,
        "--runner-abi", CAMPAIGN_TRAINING_RUNNER_ABI,
        "--runtime-binding-digest", runtime_binding_digest,
        "--runtime-release-digest", release_digest,
        "--seed", str(seed),
        "--start-confirmation-path", str(confirmation_path),
        "--start-gate-path", str(gate_path),
    ]
    if resource_telemetry:
        command.extend(
            [
                "--resource-telemetry",
                "--resource-telemetry-path",
                str(telemetry_path),
            ]
        )
    if prefix_contract_path is not None:
        command.extend(
            ["--prefix-contract-path", str(prefix_contract_path.resolve())]
        )
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        command,
        cwd=capability.run_working_directory,
        env=_worker_environment(capability, candidate_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while not confirmation_path.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            return {
                "error_message": "worker exited before START_CONFIRMED",
                "exit_status": "RUNTIME_FAILURE",
                "launcher_return_code": process.returncode,
                "stderr_digest": sha256_digest(stderr),
                "stdout_digest": sha256_digest(stdout),
                "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
            }
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            return {
                "error_message": "worker START_CONFIRMED timeout",
                "exit_status": "RUNTIME_FAILURE",
                "launcher_return_code": 124,
                "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
            }
        time.sleep(0.05)
    confirmation = _read_json(confirmation_path)
    if int(confirmation.get("pid", -1)) != process.pid:
        process.kill()
        process.wait()
        raise FreshR1Error("training START_CONFIRMED pid mismatch")
    _write_start_gate(gate_path, start_identity)
    effective_timeout = (
        min(timeout_seconds, watchdog_seconds)
        if watchdog_seconds is not None
        else timeout_seconds
    )
    censoring_trigger = None
    try:
        stdout, stderr = process.communicate(timeout=effective_timeout)
        return_code = int(process.returncode)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return_code = 124
        censoring_trigger = (
            "ENGINEERING_WATCHDOG"
            if watchdog_seconds is not None and watchdog_seconds <= timeout_seconds
            else "RESOURCE_BUDGET_DEADLINE"
        )
    wall_time_ms = max(1, (time.monotonic_ns() - started_ns) // 1_000_000)
    worker = _read_json(output_path) if output_path.is_file() else {
        "error_message": "worker result missing",
        "exit_status": "RUNTIME_FAILURE",
    }
    durable_telemetry = (
        _read_json(telemetry_path) if telemetry_path.is_file() else None
    )
    metrics = {
        str(key).lower(): float(value)
        for key, value in dict(worker.get("best_valid_result", {})).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    result = {
        "binding_digest": binding_digest,
        "device_evidence": worker.get("training_device_evidence"),
        "epochs_requested": epochs,
        "execution_recipe_digest": sha256_digest(recipe),
        "exit_status": (
            "RESOURCE_CENSORED"
            if censoring_trigger is not None
            else worker.get("exit_status")
        ),
        "filesystem_mount_audit": worker.get("filesystem_mount_audit"),
        "launcher_return_code": return_code,
        "log_sha256": bytes_sha256(log_path.read_bytes()) if log_path.is_file() else None,
        "metrics": metrics,
        "result_sha256": bytes_sha256(output_path.read_bytes()) if output_path.is_file() else None,
        "runtime_binding_digest": runtime_binding_digest,
        "runtime_release_digest": release_digest,
        "seed": seed,
        "stderr_digest": sha256_digest(stderr),
        "stdout_digest": sha256_digest(stdout),
        "wall_time_ms": wall_time_ms,
        "worker_error_message": worker.get("error_message"),
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
    if watchdog_seconds is not None:
        result.update(
            {
                "censoring_semantics": (
                    "RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT"
                    if censoring_trigger is not None
                    else None
                ),
                "censoring_trigger": censoring_trigger,
                "mechanism_effect_update_allowed": False,
                "resource_deadline_seconds": timeout_seconds,
                "watchdog_seconds": watchdog_seconds,
            }
        )
    return canonical_value(result)


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
    binding = canonical_value(
        {
            "baseline_binding_digest": baseline_run["binding_digest"],
            "candidate_binding_digest": candidate_run["binding_digest"],
            "matched_seed": candidate_run["seed"],
            "protocol_digest": spec.protocol_digest,
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
        experiment_binding_ref=(
            f"{CORRECTED_RUN_IDENTITY}-experiment-binding:{sha256_digest(binding)}"
        ),
        experiment_binding_digest=sha256_digest(binding),
        comparator_ref=(
            f"{CORRECTED_RUN_IDENTITY}-bpr-comparator:"
            f"{baseline_run['binding_digest']}"
        ),
        comparator_digest=sha256_digest(baseline_run),
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


def _shared_policy(implementation_prompt_digest: str, tool_policy_digest: str) -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/__init__.py", "recclaw_ext/candidate.py"),
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
            )
            blind_candidate_id = str(
                materialized.shared_request["blind_candidate_id"]
            )
            candidate_root = (
                side_root / "candidates" / slot_id / blind_candidate_id
            )
            candidate_source = candidate_root / "recclaw_ext/candidate.py"
            candidate_run = run_development_training(
                repo_root=repo_root,
                side_root=side_root,
                run_id=f"{slot_id}-candidate",
                seed=seed,
                candidate_root=candidate_root,
                entrypoint=materialized.package.executable_entrypoint,
                source_sha256=bytes_sha256(candidate_source.read_bytes()),
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
    "FreshR1Error",
    "ProviderAttemptResult",
    "bounded_provider_call",
    "evaluate_gate",
    "render_implementation_prompt",
    "render_proposal_prompt",
    "retry_eligible",
    "run_formal_fresh_r1",
    "verify_formal_identity",
]
