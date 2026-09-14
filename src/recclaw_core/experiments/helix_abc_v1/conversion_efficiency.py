"""Small, pre-outcome conversion helpers for the Q5-A follow-up.

The module deliberately contains no provider, metric, or training calls.  It
only describes the observed mechanical-repair boundary and the fixed
screen-to-promotion schedule consumed by the existing stage runner.
"""

from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import re
from math import ceil
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest


CONVERSION_SCHEMA = "recclaw.research-line.q5-conversion-efficiency.v1"
MECHANICAL_REPAIR_SCHEMA = (
    "recclaw.research-line.q5-mechanical-implementer-repair.v1"
)
SCREEN_EPOCHS = 20
FULL_EPOCHS = 100
MAX_REPAIR_TURNS = 1
FULL_DEVELOPMENT_SEEDS = (54304, 54305)
FULL_RESOURCE_BUDGET_SECONDS = 7200
FULL_WATCHDOG_SECONDS = 10800
RESOURCE_DEADLINE_MIN_SECONDS = 180
RESOURCE_DEADLINE_MARGIN = 1.10
SCREEN_DEADLINE_STARTUP_MARGIN_SECONDS = 60
SCREEN_DEADLINE_VARIANCE_MARGIN = 1.25
CANDIDATE_LOCAL_ALLOWED_FILES = (
    "recclaw_ext/__init__.py",
    "recclaw_ext/candidate.py",
    "recclaw_ext/layers.py",
    "recclaw_ext/modules.py",
    "recclaw_ext/ops.py",
)
RECBole_INTERFACE_CONTRACT = {
    "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
    "model_base": "recbole.model.abstract_recommender.GeneralRecommender",
    "constructor": "(config, dataset)",
    "required_methods": ("calculate_loss", "predict", "full_sort_predict"),
    "gpu_budget_gb": 10,
    "candidate_package": {
        "import_root": "candidate_root",
        "package": "recclaw_ext",
        "required_files": (
            "recclaw_ext/__init__.py",
            "recclaw_ext/candidate.py",
        ),
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "absolute_imports": "candidate-local recclaw_ext.* modules",
    },
}
_SEALED_PARENT_ATTRIBUTES = frozenset(
    {
        "USER_ID",
        "ITEM_ID",
        "NEG_ITEM_ID",
        "n_users",
        "n_items",
        "device",
        "training",
        "logger",
    }
)
_SEALED_PARENT_METHODS = frozenset(
    {
        "add_module",
        "apply",
        "eval",
        "forward",
        "full_sort_predict",
        "load_other_parameter",
        "named_parameters",
        "other_parameter",
        "parameters",
        "predict",
        "register_buffer",
        "register_parameter",
        "state_dict",
        "train",
    }
)
MECHANICAL_SHAPE_TEST_HINT = {
    "calculate_loss": {"user_id": [4], "item_id": [4], "neg_item_id": [4]},
    "predict": {"user_id": [4], "item_id": [4]},
    "full_sort_predict": {"user_id": [4], "candidate_count": 32},
    "complexity_hint": "avoid full_catalog_by_embedding_dim_by_embedding_dim intermediates",
}
MECHANICAL_REPAIR_STAGES = frozenset(
    {
        "SCHEMA",
        "STATIC_VALIDATION",
        "CONSTRUCTION",
        "API_CONTRACT",
        "UNIT",
        "ONE_EPOCH_SMOKE",
        "RESOURCE_PROBE",
    }
)
MECHANICAL_REPAIR_CLASSES = frozenset({"IMPLEMENTATION", "INTERFACE", "RUNTIME"})
_SOURCE_BEARING_IMPLEMENTATION_MEMORY_REASON = (
    "MEASURED_IMPLEMENTATION_MEMORY_EXCEEDS_DEVICE"
)
_DEVICE_MEMORY_RESOURCE_REASON = "CUDA_DEVICE_MEMORY_ALLOCATION_FAILED"
_DEVICE_MEMORY_ERROR_TYPES = frozenset(
    {
        "torch.cuda.outofmemoryerror",
        "torch.outofmemoryerror",
        "torch._c.outofmemoryerror",
    }
)
_DEVICE_MEMORY_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda error out of memory",
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cusparse_status_alloc_failed",
)
_DEVICE_MEMORY_PATTERNS = (
    re.compile(
        r"\b(?:cuda|gpu|device)[ -]?memory "
        r"(?:allocation )?(?:failed|failure)\b"
    ),
    re.compile(
        r"\b(?:failed|unable|cannot|can't) to allocate\b[^\n]{0,120}"
        r"\b(?:cuda|gpu|device)[ -]?memory\b"
    ),
    re.compile(
        r"\b(?:failed|unable|cannot|can't) to allocate\b[^\n]{0,120}"
        r"\bmemory\b[^\n]{0,40}\bon (?:cuda )?device\b"
    ),
    re.compile(
        r"\b(?:cuda|gpu|device)[ -]?allocator\b[^\n]{0,120}"
        r"\b(?:out of memory|allocation failed)\b"
    ),
)
_OUTCOME_WORDS = frozenset(
    {
        "effect",
        "metric",
        "ndcg",
        "outcome",
        "result",
        "winner",
        "qualification_result",
    }
)


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _OUTCOME_WORDS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_outcome(item) for item in value)
    return False


def normalize_qualification_failure(
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize explicit device-memory exhaustion without broad runtime matching."""

    normalized = dict(failure)
    if str(normalized.get("failure_class", "")).upper() == "RESOURCE":
        return normalized
    error_type = str(normalized.get("error_type", "")).strip().lower()
    evidence = "\n".join(
        str(normalized.get(field, "")).lower()
        for field in ("error_type", "reason_code", "message", "traceback")
    )
    type_matches = error_type in _DEVICE_MEMORY_ERROR_TYPES
    marker_matches = any(marker in evidence for marker in _DEVICE_MEMORY_MARKERS)
    pattern_matches = any(pattern.search(evidence) for pattern in _DEVICE_MEMORY_PATTERNS)
    if type_matches or marker_matches or pattern_matches:
        normalized["failure_class"] = "RESOURCE"
        normalized["reason_code"] = _DEVICE_MEMORY_RESOURCE_REASON
    return normalized


def is_mechanical_repair_failure(failure: Mapping[str, Any]) -> bool:
    """Return whether a qualifier/validator failure may enter one repair call."""

    if _contains_outcome(failure):
        return False
    normalized = normalize_qualification_failure(failure)
    stage = str(normalized.get("stage", "")).upper()
    failure_class = str(normalized.get("failure_class", "")).upper()
    detail = normalized.get("detail")
    detail = detail if isinstance(detail, Mapping) else {}
    failure_scope = str(
        normalized.get("failure_scope") or detail.get("failure_scope") or ""
    ).upper()
    reason_code = str(
        normalized.get("reason_code") or detail.get("reason_code") or ""
    ).upper()
    failure_phase = str(
        normalized.get("failure_phase")
        or normalized.get("phase")
        or detail.get("failure_phase")
        or detail.get("phase")
        or ""
    ).upper()
    if (
        failure_scope == "RECOVERY"
        or reason_code == "RESOURCE_PROBE_INTERRUPTED_RECOVERY"
        or failure_phase == "RESOURCE_PROBE_RECOVERY"
    ):
        # Recovery state belongs to orchestration, never to candidate source.
        # Even contaminated legacy diagnostics that happen to name a method
        # must not trigger an Implementer rewrite.
        return False
    symbol_less_efficiency_repair = (
        reason_code
        in {
            "MEASURED_IMPLEMENTATION_THROUGHPUT_EXCEEDS_ENVELOPE",
            "MEASURED_IMPLEMENTATION_THROUGHPUT_NOT_IMPROVED",
        }
        or (
            reason_code == _SOURCE_BEARING_IMPLEMENTATION_MEMORY_REASON
            and failure_scope == "CANDIDATE_LOCAL"
            and normalized.get("preserve_mechanism_program") is True
        )
    )
    if (
        stage == "RESOURCE_PROBE"
        and not (
            _implicated_methods(normalized)
            or _implicated_source_files(normalized)
        )
        and not symbol_less_efficiency_repair
    ):
        # A throughput/resource observation without candidate-owned symbols is
        # evidence for rerouting, not a legal instruction to rewrite the whole
        # implementation.  Blind resource revisions repeatedly produced
        # NO_EFFECT or mechanism/base drift in the live B/C lanes.  The two
        # typed efficiency reasons are scoped separately to changed sampler
        # entrypoints rather than authorizing a whole-package rewrite.
        return False
    return (
        stage in MECHANICAL_REPAIR_STAGES
        and failure_class in MECHANICAL_REPAIR_CLASSES
    )


def build_mechanical_repair_request(
    original_request: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    current_source: Mapping[str, str] | None = None,
    failure_message: str | None = None,
    short_trace: str | None = None,
    repair_attempt: int = 1,
) -> dict[str, Any]:
    """Bind only a mechanical failure to the same blind implementer request."""

    if not is_mechanical_repair_failure(failure):
        raise ValueError("failure is outside the mechanical repair boundary")
    if int(repair_attempt) < 1 or int(repair_attempt) > MAX_REPAIR_TURNS:
        raise ValueError("repair attempt is outside the finite revision budget")
    source = {
        str(path): str(content)
        for path, content in (current_source or {}).items()
    }
    service_policy = original_request.get("service_policy")
    execution_contract = (
        service_policy.get("execution_contract")
        if isinstance(service_policy, Mapping)
        else None
    )
    base_model_config = (
        execution_contract.get("base_model_config")
        if isinstance(execution_contract, Mapping)
        else None
    )
    # The revision request is the durable hand-off to the later scoper.  Carry
    # the qualifier's method-level relation effects into that canonical hand-off
    # so the full coherent scorer snapshot is judged against the same ABI scope.
    implicated_methods = _implicated_methods(failure)
    implicated_files = _implicated_source_files(failure)
    preserved_failure_context = {
        key: failure[key]
        for key in (
            "detail",
            "failure_scope",
            "repair_scope",
            "preserve_mechanism_program",
        )
        if key in failure
    }
    implicated_file_only_target = bool(
        not implicated_methods
        and len(implicated_files) == 1
        and (
            implicated_files[0] != "recclaw_ext/candidate.py"
            or (
                str(failure.get("stage", "")).upper() == "RESOURCE_PROBE"
                and failure.get("repair_scope")
                == "IMPLEMENTATION_RESOURCE_BEHAVIOR_ONLY"
                and failure.get("preserve_mechanism_program") is True
            )
        )
    )
    context = canonical_value(
        {
            "failure_class": str(failure["failure_class"]),
            "reason_code": str(failure.get("reason_code", "UNKNOWN")),
            "stage": str(failure["stage"]),
            "message": str(failure_message or failure.get("message", ""))[:2000],
            "short_trace": str(short_trace or failure.get("traceback", ""))[:4000],
            "current_source_files": source,
            "shape_test_hint": MECHANICAL_SHAPE_TEST_HINT,
            **preserved_failure_context,
            **(
                {"implicated_methods": implicated_methods}
                if implicated_methods
                else {}
            ),
            **(
                {"implicated_files": implicated_files}
                if implicated_files
                else {}
            ),
            **(
                {"implicated_file": failure["implicated_file"]}
                if isinstance(failure.get("implicated_file"), str)
                else {}
            ),
            **(
                {"trainer_entrypoint": failure["trainer_entrypoint"]}
                if isinstance(failure.get("trainer_entrypoint"), str)
                else {}
            ),
            **(
                {"base_model_config": base_model_config}
                if isinstance(base_model_config, str) and base_model_config
                else {}
            ),
            **(
                {
                    "repair_target": {
                        "files": implicated_files,
                        "implicated_methods": (),
                        "scope_kind": "IMPLICATED_FILE_ONLY",
                    }
                }
                if implicated_file_only_target
                else {}
            ),
        }
    )
    repaired = canonical_value(
        {
            **dict(original_request),
            "repair_context": context,
            "repair_attempt": int(repair_attempt),
            "schema": MECHANICAL_REPAIR_SCHEMA,
        }
    )
    return repaired


_SCORER_METHODS = ("predict", "full_sort_predict")
_WRONG_GENERAL_RECOMMENDER_IMPORT = (
    "from recbole.model.general_recommender import GeneralRecommender"
)
_RIGHT_GENERAL_RECOMMENDER_IMPORT = (
    "from recbole.model.abstract_recommender import GeneralRecommender"
)
_COMPILED_SOURCE_BINDING_NAMES = (
    "RECCLAW_COMPILER_CANDIDATE_ID",
    "RECCLAW_MECHANISM_PROGRAM_DIGEST",
    "RECCLAW_MECHANISM_SEMANTICS_DIGEST",
    "RECCLAW_IMPLEMENTED_COMPONENT_IDS",
    "RECCLAW_IMPLEMENTED_COMPONENT_SPECS",
    "RECCLAW_IMPLEMENTED_PRIMITIVE_IDS",
    "RECCLAW_IMPLEMENTED_CUSTOM_COMPONENT_IDS",
    "RECCLAW_IMPLEMENTED_ARCHITECTURE_OPERATOR_IDS",
)


def _local_class_lineage(tree: ast.Module, model: ast.ClassDef) -> tuple[ast.ClassDef, ...]:
    """Resolve the candidate's source-local single-inheritance chain only."""
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    lineage = []
    seen = set()
    while model.name not in seen:
        lineage.append(model)
        seen.add(model.name)
        if len(model.bases) != 1 or not isinstance(model.bases[0], ast.Name):
            break
        parent = classes.get(model.bases[0].id)
        if parent is None:
            break
        model = parent
    return tuple(lineage)


def _class_methods(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source, filename="recclaw_ext/candidate.py")
    model = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
        ),
        None,
    )
    if model is None:
        return {}
    methods = {}
    for owner in _local_class_lineage(tree, model):
        for node in owner.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.setdefault(node.name, node)
    return methods


def normalize_recbole_general_recommender_import(
    source_files: Mapping[str, str],
) -> dict[str, str]:
    """Correct the one observed frozen-RecBole import incompatibility."""

    normalized = dict(source_files)
    path = "recclaw_ext/candidate.py"
    source = normalized.get(path)
    if not isinstance(source, str):
        return normalized
    lines = source.splitlines(keepends=True)
    normalized[path] = "".join(
        (
            _RIGHT_GENERAL_RECOMMENDER_IMPORT
            + ("\n" if line.endswith("\n") else "")
        )
        if line.rstrip("\n") == _WRONG_GENERAL_RECOMMENDER_IMPORT
        else line
        for line in lines
    )
    return normalized


def normalize_candidate_runtime_imports(
    source_files: Mapping[str, str],
) -> dict[str, str]:
    """Own common runtime imports that are mechanical rather than scientific."""

    normalized = normalize_recbole_general_recommender_import(source_files)
    path = "recclaw_ext/candidate.py"
    source = normalized.get(path)
    if not isinstance(source, str):
        return normalized
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return normalized
    uses_torch = any(
        isinstance(node, ast.Name)
        and node.id == "torch"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )
    binds_torch = any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "torch" and alias.asname in {None, "torch"}
            for alias in node.names
        )
        for node in tree.body
    )
    if not uses_torch or binds_torch:
        return normalized

    insertion_line = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insertion_line = int(tree.body[0].end_lineno or 0)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = max(insertion_line, int(node.end_lineno or 0))
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.splitlines(keepends=True)
    lines.insert(insertion_line, f"import torch{newline}")
    normalized[path] = "".join(lines)
    return normalized


def _method_fingerprint(
    methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    name: str,
) -> str | None:
    node = methods.get(name)
    return None if node is None else ast.dump(node, include_attributes=False)


def _implicated_methods(failure: Mapping[str, Any]) -> tuple[str, ...]:
    values = failure.get("implicated_methods", ())
    if not isinstance(values, (tuple, list)):
        values = ()
    detail = failure.get("detail")
    detail_values = (
        detail.get("implicated_methods", ())
        if isinstance(detail, Mapping)
        else ()
    )
    if not isinstance(detail_values, (tuple, list)):
        detail_values = ()
    relation_effects = failure.get("relation_intervention_effects", {})
    effect_methods = (
        tuple(
            str(name)
            for name, changed in relation_effects.items()
            if isinstance(name, str) and name.isidentifier() and changed is True
        )
        if isinstance(relation_effects, Mapping)
        else ()
    )
    return tuple(
        dict.fromkeys(
            str(name)
            for name in (*values, *detail_values, *effect_methods)
            if isinstance(name, str) and name.isidentifier()
        )
    )


def _implicated_source_files(failure: Mapping[str, Any]) -> tuple[str, ...]:
    values = failure.get("implicated_files", ())
    if not isinstance(values, (tuple, list)):
        values = ()
    detail = failure.get("detail")
    detail_values = (
        detail.get("implicated_files", ())
        if isinstance(detail, Mapping)
        else ()
    )
    if not isinstance(detail_values, (tuple, list)):
        detail_values = ()
    explicit = failure.get("implicated_file")
    if isinstance(explicit, str):
        values = (*values, explicit)
    entrypoints = tuple(
        failure.get(field)
        for field in ("entrypoint", "trainer_entrypoint")
        if isinstance(failure.get(field), str)
    )
    paths: list[str] = []
    for value in (*values, *detail_values, *entrypoints):
        path = str(value).split(":", 1)[0]
        if "/" not in path:
            path = path.replace(".", "/")
        if path.startswith("recclaw_ext/") and not path.endswith(".py"):
            path += ".py"
        if path.startswith("recclaw_ext/") and path.endswith(".py"):
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _module_assignment_name(node: ast.stmt) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def scope_mechanical_repair_source(
    *,
    current_source: Mapping[str, str],
    repaired_source: Mapping[str, str],
    failure: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any] | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Keep a compiler-binding repair from rewriting executable source."""

    if (
        str(failure.get("reason_code", "")).upper()
        == _SOURCE_BEARING_IMPLEMENTATION_MEMORY_REASON
        and str(failure.get("failure_scope", "")).upper() == "CANDIDATE_LOCAL"
        and failure.get("preserve_mechanism_program") is True
    ):
        scoped = dict(current_source)
        for path in CANDIDATE_LOCAL_ALLOWED_FILES:
            if path in repaired_source:
                scoped[path] = str(repaired_source[path])
        validate_mechanical_repair_source_scope(
            current_source=current_source,
            repaired_source=scoped,
            failure=failure,
        )
        changed = False
        for path in set(current_source) | set(scoped):
            before = current_source.get(path)
            after = scoped.get(path)
            if before == after:
                continue
            if (
                path.endswith(".py")
                and isinstance(before, str)
                and isinstance(after, str)
            ):
                try:
                    if ast.dump(
                        ast.parse(before, filename=path),
                        include_attributes=False,
                    ) == ast.dump(
                        ast.parse(after, filename=path),
                        include_attributes=False,
                    ):
                        continue
                except SyntaxError:
                    pass
            changed = True
            break
        if not changed:
            raise ValueError(
                "MECHANICAL_REPAIR_NO_EFFECT: scoped revision is unchanged"
            )
        return scoped

    if str(failure.get("reason_code", "")) != (
        "COMPILED_MECHANISM_SOURCE_BINDING_MISMATCH"
    ):
        repair_target = failure.get("repair_target")
        implicated_file_only = bool(
            isinstance(repair_target, Mapping)
            and repair_target.get("scope_kind") == "IMPLICATED_FILE_ONLY"
        )
        scoped = (
            dict(current_source)
            if implicated_file_only
            else _compose_implicated_candidate_symbols(
                current_source=current_source,
                repaired_source=repaired_source,
                failure=failure,
                exact_parent_bundle=exact_parent_bundle,
            )
        )
        if implicated_file_only and "recclaw_ext/candidate.py" in (
            _implicated_source_files(failure)
        ):
            # A measured whole-file efficiency request is not a method-local
            # exception repair. Keep its coherent implementation, then apply
            # the existing scorer and materialization/qualification contracts.
            replacement = repaired_source.get("recclaw_ext/candidate.py")
            if not isinstance(replacement, str):
                raise ValueError("mechanical repair must preserve recclaw_ext/candidate.py")
            scoped["recclaw_ext/candidate.py"] = replacement
        validate_mechanical_repair_source_scope(
            current_source=current_source,
            repaired_source=scoped,
            failure=failure,
        )
        implicated_files = set(_implicated_source_files(failure))
        scoped_revision = dict(current_source)
        scoped_revision["recclaw_ext/candidate.py"] = scoped[
            "recclaw_ext/candidate.py"
        ]
        for path in set(scoped) - set(current_source):
            scoped_revision[path] = scoped[path]
        for path in implicated_files - {"recclaw_ext/candidate.py"}:
            if path in repaired_source:
                scoped_revision[path] = repaired_source[path]
            else:
                scoped_revision.pop(path, None)
        changed = False
        for path in set(current_source) | set(scoped_revision):
            before = current_source.get(path)
            after = scoped_revision.get(path)
            if before == after:
                continue
            if (
                path.endswith(".py")
                and isinstance(before, str)
                and isinstance(after, str)
            ):
                try:
                    if ast.dump(
                        ast.parse(before, filename=path),
                        include_attributes=False,
                    ) == ast.dump(
                        ast.parse(after, filename=path),
                        include_attributes=False,
                    ):
                        continue
                except SyntaxError:
                    pass
            changed = True
            break
        if not changed:
            raise ValueError(
                "MECHANICAL_REPAIR_NO_EFFECT: scoped revision is unchanged"
            )
        return scoped_revision

    path = "recclaw_ext/candidate.py"
    current = current_source.get(path)
    repaired = repaired_source.get(path)
    if not isinstance(current, str) or not isinstance(repaired, str):
        raise ValueError("compiler-binding repair must preserve candidate.py")
    try:
        current_tree = ast.parse(current, filename=path)
        repaired_tree = ast.parse(repaired, filename=path)
    except SyntaxError as error:
        raise ValueError("compiler-binding repair source is not valid Python") from error

    if compiled_mechanism is None:
        repaired_bindings = {
            name: node
            for node in repaired_tree.body
            if (name := _module_assignment_name(node))
            in _COMPILED_SOURCE_BINDING_NAMES
        }
    else:
        implementation_binding = compiled_mechanism.get("implementation_binding")
        if not isinstance(implementation_binding, Mapping):
            raise ValueError("compiled mechanism lacks implementation_binding")
        exact_values = {
            "RECCLAW_COMPILER_CANDIDATE_ID": compiled_mechanism.get(
                "compiler_candidate_id"
            ),
            "RECCLAW_MECHANISM_PROGRAM_DIGEST": compiled_mechanism.get(
                "mechanism_program_digest"
            ),
            "RECCLAW_MECHANISM_SEMANTICS_DIGEST": compiled_mechanism.get(
                "mechanism_semantics_digest"
            ),
            "RECCLAW_IMPLEMENTED_COMPONENT_IDS": implementation_binding.get(
                "component_ids"
            ),
            "RECCLAW_IMPLEMENTED_COMPONENT_SPECS": implementation_binding.get(
                "component_specs"
            ),
            "RECCLAW_IMPLEMENTED_PRIMITIVE_IDS": implementation_binding.get(
                "primitive_ids"
            ),
            "RECCLAW_IMPLEMENTED_CUSTOM_COMPONENT_IDS": (
                implementation_binding.get("custom_component_ids")
            ),
            "RECCLAW_IMPLEMENTED_ARCHITECTURE_OPERATOR_IDS": (
                implementation_binding.get("architecture_operator_ids")
            ),
        }
        if any(value is None for value in exact_values.values()):
            raise ValueError("compiled mechanism lacks an exact source binding")
        repaired_bindings = {
            name: ast.parse(
                f"{name} = {canonical_value(value)!r}",
                filename=path,
            ).body[0]
            for name, value in exact_values.items()
        }
    if not repaired_bindings:
        raise ValueError("compiler-binding repair supplied no binding declarations")

    seen: set[str] = set()
    body: list[ast.stmt] = []
    for node in current_tree.body:
        name = _module_assignment_name(node)
        if name in repaired_bindings:
            body.append(deepcopy(repaired_bindings[name]))
            seen.add(name)
        else:
            body.append(node)
    missing = [
        deepcopy(repaired_bindings[name])
        for name in _COMPILED_SOURCE_BINDING_NAMES
        if name in repaired_bindings and name not in seen
    ]
    insert_at = next(
        (index for index, node in enumerate(body) if isinstance(node, ast.ClassDef)),
        len(body),
    )
    body[insert_at:insert_at] = missing
    current_tree.body = body
    ast.fix_missing_locations(current_tree)

    scoped = dict(current_source)
    scoped[path] = ast.unparse(current_tree) + "\n"
    validate_mechanical_repair_source_scope(
        current_source=current_source,
        repaired_source=scoped,
        failure=failure,
    )
    return scoped


def _restore_failure_unrelated_scorer_methods(
    *,
    current_source: Mapping[str, str],
    repaired_source: Mapping[str, str],
    failure: Mapping[str, Any],
) -> dict[str, str]:
    """Keep the original scorer while accepting an unrelated mechanical fix."""

    path = "recclaw_ext/candidate.py"
    current = current_source.get(path)
    repaired = repaired_source.get(path)
    if not isinstance(current, str) or not isinstance(repaired, str):
        raise ValueError("mechanical repair must preserve recclaw_ext/candidate.py")
    try:
        current_tree = ast.parse(current, filename=path)
        repaired_tree = ast.parse(repaired, filename=path)
    except SyntaxError as error:
        raise ValueError("mechanical repair scorer scope is not valid Python") from error
    current_model = next(
        (
            node
            for node in current_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
        ),
        None,
    )
    repaired_model = next(
        (
            node
            for node in repaired_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
        ),
        None,
    )
    if current_model is None or repaired_model is None:
        return dict(repaired_source)
    implicated_methods = set(_implicated_methods(failure))
    evidence = "\n".join(
        str(failure.get(field, "")).lower()
        for field in (
            "stage",
            "failure_class",
            "reason_code",
            "message",
            "short_trace",
            "traceback",
        )
    )
    current_methods = {
        node.name: node
        for node in current_model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    changed = False
    for method in _SCORER_METHODS:
        if (
            method in implicated_methods
            or (not implicated_methods and method.lower() in evidence)
            or method not in current_methods
        ):
            continue
        replacement = deepcopy(current_methods[method])
        for index, node in enumerate(repaired_model.body):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method
            ):
                if ast.dump(node, include_attributes=False) != ast.dump(
                    replacement, include_attributes=False
                ):
                    repaired_model.body[index] = replacement
                    changed = True
                break
        else:
            repaired_model.body.append(replacement)
            changed = True
    if not changed:
        return dict(repaired_source)
    ast.fix_missing_locations(repaired_tree)
    scoped = dict(repaired_source)
    scoped[path] = ast.unparse(repaired_tree) + "\n"
    return scoped


def _compose_implicated_candidate_symbols(
    *,
    current_source: Mapping[str, str],
    repaired_source: Mapping[str, str],
    failure: Mapping[str, Any],
    exact_parent_bundle: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Transplant one implicated method closure and its missing dependencies."""

    path = "recclaw_ext/candidate.py"
    current = current_source.get(path)
    repaired = repaired_source.get(path)
    if not isinstance(current, str) or not isinstance(repaired, str):
        raise ValueError("mechanical repair must preserve recclaw_ext/candidate.py")
    try:
        current_tree = ast.parse(current, filename=path)
        repaired_tree = ast.parse(repaired, filename=path)
    except SyntaxError as error:
        raise ValueError("mechanical repair source is not valid Python") from error
    current_model = next(
        (node for node in current_tree.body if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"),
        None,
    )
    repaired_model = next(
        (node for node in repaired_tree.body if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"),
        None,
    )
    if current_model is None or repaired_model is None:
        return dict(current_source)
    # Generated candidates may put the implementation in a local parent and
    # expose FreshCandidateModel as a thin entrypoint. Repair the actual owner;
    # do not discard its revision or transplant it into a different class.
    current_lineage = _local_class_lineage(current_tree, current_model)
    repaired_lineage = _local_class_lineage(repaired_tree, repaired_model)
    repair_roots = set(_implicated_methods(failure))
    if not repair_roots:
        evidence = "\n".join(
            str(failure.get(field, "")).lower()
            for field in ("stage", "failure_class", "reason_code", "message", "short_trace", "traceback")
        )
        repair_roots = {
            node.name for owner in repaired_lineage for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lower() in evidence
        }
    entrypoint_methods = {
        node.name for owner in (current_model, repaired_model) for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not repair_roots.intersection(entrypoint_methods):
        for owner in current_lineage[1:]:
            owned_methods = {
                node.name for node in owner.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if repair_roots.intersection(owned_methods):
                revised_owner = next((node for node in repaired_lineage if node.name == owner.name), None)
                if revised_owner is None:
                    raise ValueError("MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: repaired source omits implicated owner")
                current_model, repaired_model = owner, revised_owner
                break
    current_methods = {
        node.name: node for node in current_model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    repaired_methods = {
        node.name: node for node in repaired_model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Hook fragments initialize candidate state through the same explicit ABI
    # used by the parent binder. Reuse constructor transactions for that owner.
    initializer_name = (
        "recclaw_initialize_mechanism"
        if "recclaw_initialize_mechanism" in repair_roots
        or (
            "__init__" not in current_methods
            and "__init__" not in repaired_methods
            and "recclaw_initialize_mechanism" in (current_methods.keys() | repaired_methods.keys())
        )
        else "__init__"
    )
    # Qualifier stack symbols also name module functions. Use the same closure
    # and caller-contract composition when that is the actual failed owner.
    current_module_functions = {
        node.name: node for node in current_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    repaired_module_functions = {
        node.name: node for node in repaired_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module_function_names = (
        current_module_functions.keys() | repaired_module_functions.keys()
    ) - (current_methods.keys() | repaired_methods.keys())
    if repair_roots & module_function_names:
        current_methods.update({name: node for name, node in current_module_functions.items() if name in module_function_names})
        repaired_methods.update({name: node for name, node in repaired_module_functions.items() if name in module_function_names})
    else:
        module_function_names = set()
    explicitly_implicated = set(_implicated_methods(failure))
    implicated = set(explicitly_implicated)
    implicated_files = set(_implicated_source_files(failure))
    if not implicated:
        evidence = "\n".join(
            str(failure.get(field, "")).lower()
            for field in ("stage", "failure_class", "reason_code", "message", "short_trace", "traceback")
        )
        implicated.update(name for name in repaired_methods if name.lower() in evidence)
    def candidate_member_name(node: ast.Attribute) -> str | None:
        owner = node.value
        if isinstance(owner, ast.Name) and owner.id in {
            "self",
            "cls",
            "FreshCandidateModel",
            current_model.name,
        }:
            return node.attr
        if (
            isinstance(owner, ast.Call)
            and isinstance(owner.func, ast.Name)
            and owner.func.id == "type"
            and len(owner.args) == 1
            and isinstance(owner.args[0], ast.Name)
            and owner.args[0].id == "self"
        ):
            return node.attr
        return None

    def method_dependencies(
        node: ast.AST,
        methods: Mapping[str, ast.AST],
    ) -> set[str]:
        dependencies: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in module_function_names:
                dependencies.add(child.id)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and len(child.args) >= 2
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id in {"self", "cls"}
            ):
                if not (
                    isinstance(child.args[1], ast.Constant)
                    and isinstance(child.args[1].value, str)
                ):
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: dynamic "
                        "candidate member lookup cannot be composed"
                    )
                if child.args[1].value in methods:
                    dependencies.add(child.args[1].value)
                continue
            if not isinstance(child, ast.Attribute):
                continue
            name = candidate_member_name(child)
            if name in methods:
                dependencies.add(name)
        return dependencies

    def method_closure(
        names: set[str],
        methods: Mapping[str, ast.AST] = repaired_methods,
    ) -> set[str]:
        closure = {name for name in names if name in methods}
        pending = list(closure)
        while pending:
            dependencies = method_dependencies(methods[pending.pop()], methods)
            for name in dependencies:
                if name in methods and name not in closure:
                    closure.add(name)
                    pending.append(name)
        return closure

    def method_interface_fingerprint(node: ast.AST | None) -> tuple[Any, ...] | None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        args = node.args
        return (
            len(args.posonlyargs),
            len(args.args),
            args.vararg is not None,
            tuple(argument.arg for argument in args.kwonlyargs),
            tuple(default is None for default in args.kw_defaults),
            args.kwarg is not None,
            len(args.defaults),
        )

    def direct_candidate_call_nodes(
        node: ast.AST | None,
        callee: str,
    ) -> tuple[ast.Call, ...]:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ()
        calls: list[ast.Call] = []

        class DirectCallCollector(ast.NodeVisitor):
            def visit_Call(self, child: ast.Call) -> None:
                if (
                    isinstance(child.func, ast.Attribute)
                    and candidate_member_name(child.func) == callee
                    or isinstance(child.func, ast.Name)
                    and child.func.id == callee
                    and callee in module_function_names
                ):
                    calls.append(child)
                self.generic_visit(child)

            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                return

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, child: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, child: ast.Lambda) -> None:
                return

        collector = DirectCallCollector()
        for statement in node.body:
            collector.visit(statement)
        return tuple(calls)

    def direct_candidate_calls(node: ast.AST | None, callee: str) -> tuple[str, ...]:
        return tuple(
            ast.dump(call, include_attributes=False)
            for call in direct_candidate_call_nodes(node, callee)
        )

    def call_is_compatible(
        call: ast.Call,
        method: ast.AST | None,
    ) -> bool:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            return True
        if any(keyword.arg is None for keyword in call.keywords):
            return True
        positional = [*method.args.posonlyargs, *method.args.args]
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        provided_positional = len(call.args)
        if method.args.vararg is None and provided_positional > len(positional):
            return False
        assigned_names = {
            argument.arg for argument in positional[:provided_positional]
        }
        keyword_names = {
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        }
        if assigned_names & keyword_names:
            return False
        allowed_keyword_names = {
            argument.arg for argument in method.args.args
        } | {argument.arg for argument in method.args.kwonlyargs}
        if method.args.args and method.args.args[0].arg in {"self", "cls"}:
            allowed_keyword_names.discard(method.args.args[0].arg)
        if method.args.kwarg is None and keyword_names - allowed_keyword_names:
            return False
        required_positional_count = len(positional) - len(method.args.defaults)
        for argument in positional[:required_positional_count]:
            if argument.arg not in assigned_names and argument.arg not in keyword_names:
                return False
        for argument, default in zip(
            method.args.kwonlyargs,
            method.args.kw_defaults,
        ):
            if default is None and argument.arg not in keyword_names:
                return False
        return True

    missing_implicated = {
        name
        for name in implicated
        if name in current_methods and name not in repaired_methods
    }
    if missing_implicated:
        raise ValueError(
            "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: repaired source omits "
            f"implicated methods {sorted(missing_implicated)}"
        )

    selected = method_closure(implicated - {initializer_name})
    if initializer_name in implicated and initializer_name in repaired_methods:
        # Constructor ownership is composed below as a statement transaction.
        # Treating it as an ordinary method root would pull every helper called
        # anywhere in the repaired constructor into an unrelated repair.
        selected.add(initializer_name)
    if not selected and (
        str(failure.get("stage", "")).upper() == "RESOURCE_PROBE"
        and str(failure.get("reason_code", ""))
        in {
            "MEASURED_IMPLEMENTATION_THROUGHPUT_EXCEEDS_ENVELOPE",
            "MEASURED_IMPLEMENTATION_THROUGHPUT_NOT_IMPROVED",
        }
    ):
        # Epoch-sampler performance repairs have a stable compiler-owned
        # lifecycle even when the resource probe cannot name a Python method.
        # Transplant only the sampler entrypoints the revision actually changed;
        # their helper/init dependency closure is composed below while scorer,
        # objective, trainer, and unrelated package bytes remain current.
        sampler_roots = {
            name
            for name in ("recclaw_sampler_refresh", "recclaw_sampler_step")
            if name in repaired_methods
            and ast.dump(repaired_methods[name], include_attributes=False)
            != (
                ast.dump(current_methods[name], include_attributes=False)
                if name in current_methods
                else None
            )
        }
        selected = method_closure(sampler_roots)
    if not selected:
        return dict(current_source)

    # A repaired callee is not coherent when its direct caller retains an old
    # argument contract. Pull in only callers whose repaired call expression
    # differs (or whose callee signature changed), then transplant each caller
    # atomically. Method names and candidate families remain source-owned.
    direct_caller_replacements: set[str] = set()
    pending_contracts = [
        name
        for name in selected
        if name in repaired_methods
        and (
            name not in current_methods
            or ast.dump(repaired_methods[name], include_attributes=False)
            != ast.dump(current_methods[name], include_attributes=False)
        )
    ]
    visited_contracts: set[str] = set()
    while pending_contracts:
        callee = pending_contracts.pop()
        if callee in visited_contracts:
            continue
        visited_contracts.add(callee)
        signature_changed = method_interface_fingerprint(
            current_methods.get(callee)
        ) != method_interface_fingerprint(repaired_methods.get(callee))
        for caller in set(current_methods) | set(repaired_methods):
            if caller == callee:
                continue
            current_caller = current_methods.get(caller)
            repaired_caller = repaired_methods.get(caller)
            current_call_nodes = direct_candidate_call_nodes(current_caller, callee)
            current_calls = direct_candidate_calls(current_caller, callee)
            repaired_calls = direct_candidate_calls(repaired_caller, callee)
            if not current_calls and not repaired_calls:
                continue
            if not signature_changed and repaired_calls == current_calls:
                continue
            current_calls_remain_valid = all(
                call_is_compatible(call, repaired_methods.get(callee))
                for call in current_call_nodes
            )
            semantic_contract_root = (
                callee in explicitly_implicated
                or callee in direct_caller_replacements
            )
            if current_calls_remain_valid and not (
                semantic_contract_root and repaired_calls
            ):
                continue
            if repaired_caller is None:
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: repaired "
                    f"source omits caller {caller!r} for changed callee {callee!r}"
                )
            direct_caller_replacements.add(caller)
            if caller == initializer_name:
                selected.add(caller)
            else:
                selected.update(method_closure({caller}))
            if method_interface_fingerprint(current_caller) != (
                method_interface_fingerprint(repaired_caller)
            ):
                pending_contracts.append(caller)

    init_transaction_requested = (
        initializer_name in explicitly_implicated
        or initializer_name in direct_caller_replacements
    )
    replace_init_whole = (
        init_transaction_requested and initializer_name not in current_methods
    )
    if replace_init_whole and initializer_name in repaired_methods:
        selected.update(
            method_closure(
                method_dependencies(repaired_methods[initializer_name], repaired_methods)
            )
        )
    transactional_init = (
        init_transaction_requested
        and initializer_name in current_methods
        and initializer_name in repaired_methods
    )

    method_names = set(current_methods) | set(repaired_methods)

    def self_loads(node: ast.AST) -> set[str]:
        return {
            name
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
            and (name := candidate_member_name(child)) is not None
            and isinstance(child.ctx, ast.Load)
            and name not in method_names
        }

    def self_assignments(node: ast.AST) -> set[str]:
        assigned: set[str] = set()
        for child in ast.walk(node):
            targets: tuple[ast.AST, ...]
            if isinstance(child, ast.Assign):
                targets = tuple(child.targets)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
                targets = (child.target,)
            else:
                targets = ()
            for target in targets:
                for nested in ast.walk(target):
                    if (
                        isinstance(nested, ast.Attribute)
                        and isinstance(nested.value, ast.Name)
                        and nested.value.id == "self"
                    ):
                        assigned.add(nested.attr)
            if not isinstance(child, ast.Call):
                continue
            if (
                isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "self"
                and child.func.attr in {"register_buffer", "register_parameter"}
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                assigned.add(child.args[0].value)
            if (
                isinstance(child.func, ast.Name)
                and child.func.id == "setattr"
                and len(child.args) >= 2
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id == "self"
                and isinstance(child.args[1], ast.Constant)
                and isinstance(child.args[1].value, str)
            ):
                assigned.add(child.args[1].value)
        return assigned

    def assignment_bound_names(node: ast.stmt) -> set[str]:
        targets: tuple[ast.AST, ...]
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = (node.target,)
        else:
            return set()
        return {
            nested.id
            for target in targets
            for nested in ast.walk(target)
            if isinstance(nested, ast.Name) and isinstance(nested.ctx, ast.Store)
        }

    def project_binding_statement(
        node: ast.stmt,
        required_names: set[str],
    ) -> list[ast.stmt]:
        """Project only mechanically separable requested bindings."""

        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                return [deepcopy(node)] if node.targets[0].id in required_names else []
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(node.targets[0].elts) == len(node.value.elts)
                and all(isinstance(target, ast.Name) for target in node.targets[0].elts)
            ):
                return [
                    ast.Assign(targets=[deepcopy(target)], value=deepcopy(value))
                    for target, value in zip(
                        node.targets[0].elts,
                        node.value.elts,
                    )
                    if target.id in required_names
                ]
            if len(node.targets) > 1 and all(
                isinstance(target, ast.Name) for target in node.targets
            ):
                targets = [
                    deepcopy(target)
                    for target in node.targets
                    if target.id in required_names
                ]
                return (
                    [ast.Assign(targets=targets, value=deepcopy(node.value))]
                    if targets
                    else []
                )
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [deepcopy(node)] if node.target.id in required_names else []
        if assignment_bound_names(node) & required_names:
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required "
                "multi-binding statement cannot be projected"
            )
        return []

    def projected_binding_dump(node: ast.stmt, name: str) -> tuple[str, ...]:
        return tuple(
            ast.dump(projected, include_attributes=False)
            for projected in project_binding_statement(node, {name})
        )

    current_init = current_methods.get(initializer_name)
    repaired_init = repaired_methods.get(initializer_name)
    local_constructor_dependencies: set[str] = set()
    if (
        transactional_init
        and current_init is not None
        and repaired_init is not None
        and method_interface_fingerprint(current_init)
        == method_interface_fingerprint(repaired_init)
        and len(current_init.body) == len(repaired_init.body)
    ):
        changed_rows = [
            (before, after)
            for before, after in zip(current_init.body, repaired_init.body)
            if ast.dump(before, include_attributes=False)
            != ast.dump(after, include_attributes=False)
        ]
        if len(changed_rows) == 1:
            before, after = changed_rows[0]
            before_targets = (
                before.targets if isinstance(before, ast.Assign)
                else [before.target] if isinstance(before, ast.AnnAssign) else []
            )
            after_targets = (
                after.targets if isinstance(after, ast.Assign)
                else [after.target] if isinstance(after, ast.AnnAssign) else []
            )
            if (
                len(before_targets) == len(after_targets) == 1
                and isinstance(before_targets[0], ast.Name)
                and isinstance(after_targets[0], ast.Name)
                and before_targets[0].id == after_targets[0].id
                and before.value is not None
                and after.value is not None
            ):
                # A single local allocation repair is already unambiguous.
                # Keep its original constructor position and consumers instead
                # of requiring a changed self-attribute producer. Other methods
                # remain scoped normally; do not import their revisions merely
                # because the unchanged constructor also calls them.
                replace_init_whole = True
                transactional_init = False
                local_constructor_dependencies = self_loads(after) - self_loads(before)
                selected.update(method_closure(
                    method_dependencies(after, repaired_methods)
                    - method_dependencies(before, current_methods)
                ))
    if current_init is None and repaired_init is not None and len(current_lineage) > 1:
        required_new_state = {
            attribute for name in selected for attribute in self_loads(repaired_methods[name])
        }
        initialized_state = self_assignments(repaired_init)
        if initialized_state and initialized_state <= required_new_state:
            # Adding an override of an inherited constructor is atomic. Taking
            # only assignments can omit super() and the loops that populate
            # required state, producing an importable but broken candidate.
            replace_init_whole = True
            selected.add(initializer_name)
            selected.update(method_closure(method_dependencies(repaired_init, repaired_methods)))
    current_class_assignments: dict[str, ast.stmt] = {}
    repaired_class_assignments: dict[str, ast.stmt] = {}
    current_class_binding_counts: dict[str, int] = {}
    repaired_class_binding_counts: dict[str, int] = {}
    for node in current_model.body:
        for name in assignment_bound_names(node):
            current_class_assignments[name] = node
            current_class_binding_counts[name] = (
                current_class_binding_counts.get(name, 0) + 1
            )
    for node in repaired_model.body:
        for name in assignment_bound_names(node):
            repaired_class_assignments[name] = node
            repaired_class_binding_counts[name] = (
                repaired_class_binding_counts.get(name, 0) + 1
            )
    available_attributes = set(current_class_assignments)
    binder_names = {
        alias.asname or alias.name
        for node in current_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold"
        for alias in node.names
        if alias.name == "bind_epoch_sampler_model_class"
    }
    runtime_bound = any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "FreshCandidateModel"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in binder_names
        and len(node.value.args) == 2
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "FreshCandidateModel"
        and not node.value.keywords
        for node in current_tree.body
    )
    if runtime_bound:
        # The compiler wrapper supplies the exact Dataset before original_init
        # and restores it afterwards. Include it in constructor transactions
        # as well as later method dependency checks.
        available_attributes.add("dataset")
    if current_init is not None:
        available_attributes.update(self_assignments(current_init))
    elif replace_init_whole and repaired_init is not None:
        available_attributes.update(self_assignments(repaired_init))

    repaired_init_statements: dict[str, ast.stmt] = {}
    current_init_statements_by_attribute: dict[str, ast.stmt] = {}
    if current_init is not None:
        for statement in current_init.body:
            for attribute in self_assignments(statement):
                current_init_statements_by_attribute.setdefault(attribute, statement)
    if repaired_init is not None:
        for statement in repaired_init.body:
            for attribute in self_assignments(statement):
                repaired_init_statements.setdefault(attribute, statement)

    constructor_transaction_current_index: int | None = None
    constructor_transaction_current_indices: set[int] = set()
    constructor_transaction_repaired_indices: set[int] = set()
    constructor_transaction_attributes: set[str] = set()
    consumer_statement_transactions: dict[str, tuple[int, ast.stmt]] = {}
    if transactional_init:
        assert current_init is not None and repaired_init is not None
        if method_interface_fingerprint(current_init) != method_interface_fingerprint(
            repaired_init
        ):
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                "transaction changed the constructor interface"
            )

        def registered_state_name(statement: ast.stmt) -> str | None:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and call.func.attr in {"register_buffer", "register_parameter"}
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                return call.args[0].value
            return None

        def plain_self_assignment(statement: ast.stmt) -> str | None:
            registered = registered_state_name(statement)
            if registered is not None:
                return registered
            target: ast.AST | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target = statement.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return target.attr
            return None

        def state_loads(node: ast.AST, *, reject_dynamic: bool = False) -> set[str]:
            loads = set(self_loads(node)) - _SEALED_PARENT_METHODS
            for child in ast.walk(node):
                if not (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "getattr"
                    and len(child.args) >= 2
                    and isinstance(child.args[0], ast.Name)
                    and child.args[0].id in {"self", "cls"}
                ):
                    continue
                if not (
                    isinstance(child.args[1], ast.Constant)
                    and isinstance(child.args[1].value, str)
                ):
                    if reject_dynamic:
                        raise ValueError(
                            "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: dynamic "
                            "constructor-state consumer cannot be composed"
                        )
                    continue
                loads.add(child.args[1].value)
            return loads

        current_plain_rows: dict[str, list[tuple[int, ast.stmt]]] = {}
        repaired_plain_rows: dict[str, list[tuple[int, ast.stmt]]] = {}
        for index, statement in enumerate(current_init.body):
            attribute = plain_self_assignment(statement)
            if attribute is not None:
                current_plain_rows.setdefault(attribute, []).append((index, statement))
        for index, statement in enumerate(repaired_init.body):
            attribute = plain_self_assignment(statement)
            if attribute is not None:
                repaired_plain_rows.setdefault(attribute, []).append((index, statement))

        changed_attributes = {
            attribute
            for attribute, rows in current_plain_rows.items()
            if len(rows) != 1
            or len(repaired_plain_rows.get(attribute, ())) != 1
            or ast.dump(rows[0][1], include_attributes=False)
            != ast.dump(
                repaired_plain_rows[attribute][0][1],
                include_attributes=False,
            )
        }
        if not changed_attributes:
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: implicated "
                "constructor has no unique changed state producer"
            )

        def evidence_values(value: Any) -> list[str]:
            if isinstance(value, Mapping):
                return [
                    item
                    for nested in value.values()
                    for item in evidence_values(nested)
                ]
            if isinstance(value, (tuple, list)):
                return [item for nested in value for item in evidence_values(nested)]
            return [str(value)]

        evidence_text = "\n".join(
            item
            for field in (
                "stage",
                "failure_class",
                "reason_code",
                "message",
                "short_trace",
                "traceback",
                "detail",
            )
            for item in evidence_values(failure.get(field, ""))
        )
        evidence_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", evidence_text)
            if token not in {"self", "cls", initializer_name, "AttributeError"}
        }

        def producer_evidence_score(attribute: str) -> int:
            nodes = [statement for _index, statement in current_plain_rows[attribute]]
            nodes.extend(
                statement
                for _index, statement in repaired_plain_rows.get(attribute, ())
            )
            symbols = {
                child.id
                for node in nodes
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            } | {
                child.attr
                for node in nodes
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            return max(
                (len(token) for token in evidence_tokens & symbols),
                default=0,
            )

        evidence_scores = {
            attribute: producer_evidence_score(attribute)
            for attribute in changed_attributes
        }
        highest_score = max(evidence_scores.values(), default=0)
        anchors = {
            attribute
            for attribute, score in evidence_scores.items()
            if score == highest_score and score > 0
        }
        if not anchors:
            direct_contract_methods = selected - {initializer_name}
            anchors = {
                attribute
                for attribute in changed_attributes
                if any(
                    (
                        method_dependencies(statement, current_methods)
                        | method_dependencies(statement, repaired_methods)
                    )
                    & direct_contract_methods
                    for _index, statement in (
                        *current_plain_rows[attribute],
                        *repaired_plain_rows.get(attribute, ()),
                    )
                )
            }
        if not anchors:
            structurally_corresponding = {
                attribute
                for attribute in changed_attributes
                if len(current_plain_rows.get(attribute, ())) == 1
                and len(repaired_plain_rows.get(attribute, ())) == 1
            }
            if len(structurally_corresponding) == 1:
                anchors = structurally_corresponding
        if len(anchors) != 1:
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                "transaction does not have one failure-anchored state producer "
                f"{sorted(anchors or changed_attributes)}"
            )

        transaction_attribute = anchors.pop()
        current_rows = current_plain_rows.get(transaction_attribute, ())
        repaired_rows = repaired_plain_rows.get(transaction_attribute, ())
        # Assignment followed immediately by registering that same value is
        # one initialization operation. Its duplicate attribute failure can be
        # repaired with a single registration, keeping unrelated state intact.
        duplicate_registration = False
        if len(current_rows) == 2:
            (first_index, first), (last_index, last) = current_rows
            value = last.value.args[1] if registered_state_name(last) else None
            duplicate_registration = (
                last_index == first_index + 1
                and isinstance(first, (ast.Assign, ast.AnnAssign))
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
                and value.attr == transaction_attribute
            )
        if (len(current_rows) != 1 and not duplicate_registration) or len(repaired_rows) > 1:
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor state "
                f"producer {transaction_attribute!r} is not unique"
            )
        constructor_transaction_current_index = current_rows[0][0]
        constructor_transaction_current_indices = {index for index, _ in current_rows}

        if repaired_rows:
            repaired_index, repaired_statement = repaired_rows[0]
            if self_assignments(repaired_statement) != {transaction_attribute}:
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                    "replacement mutates unrelated state"
                )
            constructor_transaction_repaired_indices.add(repaired_index)
            constructor_transaction_attributes.add(transaction_attribute)
            selected.update(
                method_closure(
                    method_dependencies(repaired_statement, repaired_methods)
                )
            )
        else:
            consumers: set[str] = set()
            for name, method in current_methods.items():
                if name == initializer_name:
                    continue
                loads = state_loads(method, reject_dynamic=True)
                if transaction_attribute in loads:
                    consumers.add(name)
            for name in consumers:
                repaired_consumer = repaired_methods.get(name)
                if (
                    repaired_consumer is None
                    or method_interface_fingerprint(current_methods[name])
                    != method_interface_fingerprint(repaired_consumer)
                    or transaction_attribute
                    in state_loads(repaired_consumer, reject_dynamic=True)
                ):
                    raise ValueError(
                        "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired "
                        f"__init__ omits retained candidate state "
                        f"{[transaction_attribute]}"
                    )
                current_consumer = current_methods[name]
                current_rows = [
                    (index, statement)
                    for index, statement in enumerate(current_consumer.body)
                    if transaction_attribute
                    in state_loads(statement, reject_dynamic=True)
                ]
                if len(current_rows) != 1:
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: deleted "
                        "constructor state has no unique consumer statement"
                    )
                current_row_index, current_row = current_rows[0]
                current_bound_names = assignment_bound_names(current_row)
                stable_sink: tuple[str, ...] | None = (
                    ("BINDINGS", *sorted(current_bound_names))
                    if current_bound_names
                    else ("RETURN",)
                    if isinstance(current_row, ast.Return)
                    else None
                )
                if stable_sink is None:
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                        "state consumer has no stable statement sink"
                    )

                def consumer_sink(statement: ast.stmt) -> tuple[str, ...] | None:
                    bound_names = assignment_bound_names(statement)
                    if bound_names:
                        return ("BINDINGS", *sorted(bound_names))
                    if isinstance(statement, ast.Return):
                        return ("RETURN",)
                    return None

                repaired_rows = [
                    statement
                    for statement in repaired_consumer.body
                    if consumer_sink(statement) == stable_sink
                    and transaction_attribute
                    not in state_loads(statement, reject_dynamic=True)
                ]
                if len(repaired_rows) != 1:
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: deleted "
                        "constructor state has no unique repaired consumer sink"
                    )
                repaired_row = repaired_rows[0]
                if ast.dump(current_row, include_attributes=False) == ast.dump(
                    repaired_row,
                    include_attributes=False,
                ):
                    raise ValueError(
                        "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired "
                        f"consumer still depends on {[transaction_attribute]}"
                    )
                unresolved_consumer_state = (
                    state_loads(repaired_row, reject_dynamic=True)
                    - available_attributes
                    - set(_SEALED_PARENT_ATTRIBUTES)
                    - set(_SEALED_PARENT_METHODS)
                )
                if unresolved_consumer_state:
                    raise ValueError(
                        "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired "
                        "consumer row loads undefined candidate state "
                        f"{sorted(unresolved_consumer_state)}"
                    )
                selected.update(
                    method_closure(
                        {
                            dependency
                            for dependency in method_dependencies(
                                repaired_row,
                                repaired_methods,
                            )
                            if dependency not in current_methods
                        }
                    )
                )
                consumer_statement_transactions[name] = (
                    current_row_index,
                    deepcopy(repaired_row),
                )

        pending_transaction_attributes = [
            attribute
            for index in constructor_transaction_repaired_indices
            for attribute in state_loads(repaired_init.body[index])
            if attribute not in available_attributes
            and attribute not in _SEALED_PARENT_ATTRIBUTES
        ]
        while pending_transaction_attributes:
            attribute = pending_transaction_attributes.pop()
            if attribute in available_attributes:
                continue
            statement = repaired_init_statements.get(attribute)
            if statement is None:
                raise ValueError(
                    "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: constructor "
                    f"transaction loads undefined state {attribute!r}"
                )
            assigned = self_assignments(statement)
            if attribute not in assigned:
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                    f"state dependency {attribute!r} is not projectable"
                )
            repaired_index = repaired_init.body.index(statement)
            constructor_transaction_repaired_indices.add(repaired_index)
            constructor_transaction_attributes.update(assigned)
            available_attributes.update(assigned)
            selected.update(
                method_closure(method_dependencies(statement, repaired_methods))
            )
            pending_transaction_attributes.extend(
                state_loads(statement) - available_attributes
            )
    current_init_statement_dumps = {
        ast.dump(statement, include_attributes=False)
        for statement in (current_init.body if current_init is not None else ())
    }

    added_init_statement_ids: set[int] = {
        id(repaired_init.body[index])
        for index in constructor_transaction_repaired_indices
    } if repaired_init is not None else set()
    added_class_assignment_names: set[str] = set()
    original_available_attributes = set(available_attributes)

    while True:
        selected_before = set(selected)
        init_ids_before = set(added_init_statement_ids)
        required_attributes = {
            attribute
            for name in selected
            if name != initializer_name
            for attribute in self_loads(repaired_methods[name])
        } | constructor_transaction_attributes | local_constructor_dependencies
        for attribute in required_attributes:
            repaired_statement = repaired_init_statements.get(attribute)
            current_statement = current_init_statements_by_attribute.get(attribute)
            if (
                not replace_init_whole
                and repaired_statement is not None
                and current_statement is not None
                and ast.dump(repaired_statement, include_attributes=False)
                != ast.dump(current_statement, include_attributes=False)
            ):
                added_init_statement_ids.add(id(repaired_statement))
                selected.update(
                    method_closure(
                        method_dependencies(repaired_statement, repaired_methods)
                    )
                )
        pending_attributes = list(required_attributes - available_attributes)
        while pending_attributes:
            attribute = pending_attributes.pop()
            if attribute in available_attributes:
                continue
            statement = repaired_init_statements.get(attribute)
            if statement is not None:
                added_init_statement_ids.add(id(statement))
                available_attributes.update(self_assignments(statement))
                selected.update(
                    method_closure(method_dependencies(statement, repaired_methods))
                )
                pending_attributes.extend(
                    self_loads(statement) - available_attributes
                )
                continue
            class_assignment = repaired_class_assignments.get(attribute)
            if class_assignment is not None:
                added_class_assignment_names.add(attribute)
                available_attributes.add(attribute)

        newly_available = available_attributes - original_available_attributes
        if repaired_init is not None and not replace_init_whole:
            for statement in repaired_init.body:
                called_methods = method_dependencies(statement, repaired_methods)
                closure = method_closure(called_methods)
                if not closure:
                    continue
                closure_loads = {
                    attribute
                    for name in closure
                    for attribute in self_loads(repaired_methods[name])
                }
                closure_assignments = {
                    attribute
                    for name in closure
                    for attribute in self_assignments(repaired_methods[name])
                }
                if not (
                    closure_loads & newly_available
                    or closure_assignments & required_attributes
                ):
                    continue
                selected.update(closure)
                available_attributes.update(closure_assignments)
                if (
                    ast.dump(statement, include_attributes=False)
                    not in current_init_statement_dumps
                ):
                    added_init_statement_ids.add(id(statement))
        if (
            selected == selected_before
            and added_init_statement_ids == init_ids_before
        ):
            break

    def project_init_statement(statement: ast.stmt) -> list[ast.stmt]:
        assigned = self_assignments(statement)
        if assigned <= required_attributes:
            return [deepcopy(statement)]
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], (ast.Tuple, ast.List))
            and isinstance(statement.value, (ast.Tuple, ast.List))
            and len(statement.targets[0].elts) == len(statement.value.elts)
        ):
            projected = [
                ast.Assign(targets=[deepcopy(target)], value=deepcopy(value))
                for target, value in zip(
                    statement.targets[0].elts, statement.value.elts
                )
                if self_assignments(ast.Assign(targets=[target], value=value))
                & required_attributes
            ]
            if projected:
                return projected
        raise ValueError(
            "mechanical repair init dependency also mutates unrelated state"
        )

    def frozen_parent_contract() -> tuple[set[str], set[str]]:
        if tuple(
            ast.dump(base, include_attributes=False) for base in current_model.bases
        ) != tuple(
            ast.dump(base, include_attributes=False) for base in repaired_model.bases
        ):
            raise ValueError(
                "MECHANICAL_REPAIR_SCOPE_DRIFT: repaired candidate changed its base class"
            )
        attributes, methods = set(_SEALED_PARENT_ATTRIBUTES), set(_SEALED_PARENT_METHODS)
        # These members are supplied by the exact retained source, not by the
        # repair response. Callable modules and local ancestor methods are
        # ordinary inherited dependencies, not undefined candidate symbols.
        parents = list(_local_class_lineage(current_tree, current_model)[1:])
        # Profile hooks are fragments; their retained model lives in the
        # gateway's exact parent bundle, outside current_source_files.
        if exact_parent_bundle is not None:
            parent_source = next(
                row["content"] for row in exact_parent_bundle["files"]
                if row["path"] == path
            )
            parent_tree = ast.parse(parent_source, filename=path)
            parent_model = next(
                node for node in parent_tree.body
                if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
            )
            parents.extend(_local_class_lineage(parent_tree, parent_model))
        for parent in parents:
            for node in parent.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(node.name)
                    attributes.update(self_assignments(node))
                else:
                    attributes.update(self_assignments(node))
        return attributes, methods

    inherited_attributes, inherited_methods = frozen_parent_contract()
    available_attributes.update(inherited_attributes)
    available_attributes.update(inherited_methods)
    atomic_init_available_attributes = set(current_class_assignments)
    atomic_init_available_attributes.update(inherited_attributes)
    atomic_init_available_attributes.update(inherited_methods)
    if repaired_init is not None:
        atomic_init_available_attributes.update(self_assignments(repaired_init))
        init_method_closure = method_closure(
            method_dependencies(repaired_init, repaired_methods)
        )
        atomic_init_available_attributes.update(
            attribute
            for name in init_method_closure
            for attribute in self_assignments(repaired_methods[name])
        )
    dependency_method_nodes = [repaired_methods[name] for name in selected]
    dependency_method_nodes.extend(
        statement
        for statement in (repaired_init.body if repaired_init is not None else ())
        if id(statement) in added_init_statement_ids
    )
    available_attributes.update(
        attribute
        for node in dependency_method_nodes
        for attribute in self_assignments(node)
    )
    if replace_init_whole and repaired_init is not None and current_init is not None:
        retained_required_attributes = {
            attribute
            for name, method in current_methods.items()
            if name not in selected and name != initializer_name
            for attribute in self_loads(method)
        }
        omitted_retained_state = {
            attribute
            for attribute in (
                retained_required_attributes - atomic_init_available_attributes
            )
            if attribute in current_init_statements_by_attribute
        }
        if omitted_retained_state:
            raise ValueError(
                "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired __init__ "
                "omits retained candidate state "
                f"{sorted(omitted_retained_state)}"
            )
    current_dependency_method_nodes = [
        current_methods[name]
        for name in method_closure(implicated - {initializer_name}, current_methods)
    ]
    current_external_method_calls = {
        name
        for node in current_dependency_method_nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and (name := candidate_member_name(child.func)) is not None
        and name not in current_methods
    }
    current_external_attributes = {
        attribute
        for node in current_dependency_method_nodes
        for attribute in self_loads(node)
        if attribute not in available_attributes
    } | {
        child.args[1].value
        for node in current_dependency_method_nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "getattr"
        and len(child.args) >= 2
        and isinstance(child.args[0], ast.Name)
        and child.args[0].id == "self"
        and isinstance(child.args[1], ast.Constant)
        and isinstance(child.args[1].value, str)
    }
    unresolved_method_calls = {
        name
        for node in dependency_method_nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and (name := candidate_member_name(child.func)) is not None
        and name not in repaired_methods
        and name not in current_methods
        and name not in available_attributes
        and name not in current_external_method_calls
    }
    if unresolved_method_calls:
        raise ValueError(
            "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired symbols call "
            f"undefined methods {sorted(unresolved_method_calls)}"
        )
    unresolved_attributes = (
        required_attributes
        - available_attributes
        - current_external_attributes
    )
    duplicate_class_bindings = {
        attribute
        for attribute in required_attributes
        if current_class_binding_counts.get(attribute, 0) > 1
        or repaired_class_binding_counts.get(attribute, 0) > 1
    }
    if duplicate_class_bindings:
        raise ValueError(
            "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required class "
            f"bindings repeat {sorted(duplicate_class_bindings)}"
        )
    conflicting_class_bindings = {
        attribute
        for attribute in required_attributes
        if attribute in current_class_assignments
        and attribute in repaired_class_assignments
        and projected_binding_dump(current_class_assignments[attribute], attribute)
        != projected_binding_dump(repaired_class_assignments[attribute], attribute)
    }
    if conflicting_class_bindings:
        raise ValueError(
            "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required class "
            f"bindings differ {sorted(conflicting_class_bindings)}"
        )
    if unresolved_attributes:
        raise ValueError(
            "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired symbols load "
            f"undefined attributes {sorted(unresolved_attributes)}"
        )

    preserved_current_init_statements: list[ast.stmt] = []
    if (
        initializer_name in selected
        and not replace_init_whole
        and not transactional_init
        and current_init is not None
        and repaired_init is not None
    ):
        repaired_available_attributes = set(current_class_assignments)
        repaired_available_attributes.update(self_assignments(repaired_init))
        retained_required_attributes = {
            attribute
            for name, method in current_methods.items()
            if name not in selected and name != initializer_name
            for attribute in self_loads(method)
        }
        current_init_statements: dict[str, ast.stmt] = {}
        for statement in current_init.body:
            for attribute in self_assignments(statement):
                current_init_statements.setdefault(attribute, statement)
        pending_current_attributes = list(
            retained_required_attributes - repaired_available_attributes
        )
        preserved_ids: set[int] = set()
        while pending_current_attributes:
            attribute = pending_current_attributes.pop()
            if attribute in repaired_available_attributes:
                continue
            statement = current_init_statements.get(attribute)
            if statement is None:
                # The attribute may be supplied by the framework base class.
                # Only restore constructor state that the current candidate
                # itself demonstrably owned.
                continue
            preserved_ids.add(id(statement))
            repaired_available_attributes.update(self_assignments(statement))
            pending_current_attributes.extend(
                self_loads(statement) - repaired_available_attributes
            )
        repaired_statement_dumps = {
            ast.dump(statement, include_attributes=False)
            for statement in repaired_init.body
        }
        preserved_current_init_statements = [
            deepcopy(statement)
            for statement in current_init.body
            if id(statement) in preserved_ids
            and ast.dump(statement, include_attributes=False)
            not in repaired_statement_dumps
        ]

    added_init_statement_rows = [
        (index, projected)
        for index, statement in enumerate(
            repaired_init.body if repaired_init is not None else ()
        )
        if id(statement) in added_init_statement_ids
        for projected in project_init_statement(statement)
    ] if initializer_name not in selected or transactional_init else []
    added_init_statements = [
        statement for _index, statement in added_init_statement_rows
    ]
    added_class_assignments = [
        projected
        for node in repaired_model.body
        for projected in project_binding_statement(
            node,
            assignment_bound_names(node) & added_class_assignment_names,
        )
    ]

    body: list[ast.stmt] = []
    replaced: set[str] = set()
    for node in current_model.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in selected
            and not (node.name == initializer_name and transactional_init)
            and node.name not in consumer_statement_transactions
        ):
            replacement = deepcopy(repaired_methods[node.name])
            if node.name == initializer_name and preserved_current_init_statements:
                insert_at = next(
                    (
                        index
                        for index, statement in enumerate(replacement.body)
                        if isinstance(statement, ast.Return)
                    ),
                    len(replacement.body),
                )
                replacement.body[insert_at:insert_at] = deepcopy(
                    preserved_current_init_statements
                )
            body.append(replacement)
            replaced.add(node.name)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == initializer_name
            and added_init_statements
            and not transactional_init
        ):
            replacement = deepcopy(node)
            insert_at = next(
                (
                    index
                    for index, statement in enumerate(replacement.body)
                    if isinstance(statement, ast.Return)
                ),
                len(replacement.body),
            )
            replacement.body[insert_at:insert_at] = deepcopy(added_init_statements)
            body.append(replacement)
        else:
            body.append(node)
    if current_init is None and repaired_init is not None:
        replacement: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        if replace_init_whole and initializer_name in selected:
            replacement = deepcopy(repaired_init)
            replaced.add(initializer_name)
        elif added_init_statements:
            replacement = deepcopy(repaired_init)
            replacement.body = deepcopy(added_init_statements)
        if replacement is not None:
            constructor_insert_at = next(
                (
                    index
                    for index, node in enumerate(body)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                len(body),
            )
            body.insert(constructor_insert_at, replacement)
    class_insert_at = next(
        (
            index
            for index, node in enumerate(body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        len(body),
    )
    body[class_insert_at:class_insert_at] = added_class_assignments
    for node in repaired_model.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in selected
            and node.name not in replaced
            and not (node.name == initializer_name and current_init is None)
            and not (node.name == initializer_name and transactional_init)
            and node.name not in consumer_statement_transactions
        ):
            body.append(deepcopy(node))
    current_model.body = body
    for name, (row_index, repaired_row) in consumer_statement_transactions.items():
        composed_consumer = next(
            (
                node
                for node in current_model.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ),
            None,
        )
        if composed_consumer is None or row_index >= len(composed_consumer.body):
            raise ValueError(
                "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: consumer "
                "transaction lost its current statement anchor"
            )
        composed_consumer.body[row_index] = deepcopy(repaired_row)

    def module_bound_names(node: ast.stmt) -> set[str]:
        if isinstance(node, ast.Import):
            return {
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
            }
        if isinstance(node, ast.ImportFrom):
            return {
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            }
        names = assignment_bound_names(node)
        if names:
            return names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {node.name}
        return set()

    def projected_module_binding_dump(node: ast.stmt, name: str) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(
                ast.dump(ast.Import(names=[deepcopy(alias)]), include_attributes=False)
                for alias in node.names
                if (alias.asname or alias.name.split(".", 1)[0]) == name
            )
        if isinstance(node, ast.ImportFrom):
            return tuple(
                ast.dump(
                    ast.ImportFrom(
                        module=node.module,
                        names=[deepcopy(alias)],
                        level=node.level,
                    ),
                    include_attributes=False,
                )
                for alias in node.names
                if alias.name != "*" and (alias.asname or alias.name) == name
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return (
                (ast.dump(node, include_attributes=False),)
                if node.name == name
                else ()
            )
        return projected_binding_dump(node, name)

    current_module_bindings: dict[str, ast.stmt] = {}
    repaired_module_bindings: dict[str, ast.stmt] = {}
    current_module_binding_counts: dict[str, int] = {}
    repaired_module_binding_counts: dict[str, int] = {}
    for node in current_tree.body:
        if node is current_model:
            continue
        for name in module_bound_names(node):
            current_module_bindings[name] = node
            current_module_binding_counts[name] = (
                current_module_binding_counts.get(name, 0) + 1
            )
    for node in repaired_tree.body:
        if node is repaired_model:
            continue
        for name in module_bound_names(node):
            repaired_module_bindings[name] = node
            repaired_module_binding_counts[name] = (
                repaired_module_binding_counts.get(name, 0) + 1
            )

    selected_method_nodes = [
        repaired_methods[name]
        for name in selected
        if not (name == initializer_name and transactional_init)
        and name not in consumer_statement_transactions
    ]
    def external_name_loads(node: ast.AST) -> set[str]:
        external: set[str] = set()

        class BindingCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.bound: set[str] = set()
                self.global_names: set[str] = set()

            def visit_Name(self, child: ast.Name) -> None:
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    self.bound.add(child.id)

            def visit_Import(self, child: ast.Import) -> None:
                self.bound.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in child.names
                )

            def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
                self.bound.update(
                    alias.asname or alias.name
                    for alias in child.names
                    if alias.name != "*"
                )

            def visit_ExceptHandler(self, child: ast.ExceptHandler) -> None:
                if isinstance(child.name, str):
                    self.bound.add(child.name)
                self.generic_visit(child)

            def visit_Global(self, child: ast.Global) -> None:
                self.global_names.update(child.names)

            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                self.bound.add(child.name)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, child: ast.ClassDef) -> None:
                self.bound.add(child.name)

            def visit_Lambda(self, child: ast.Lambda) -> None:
                return

            def visit_ListComp(self, child: ast.ListComp) -> None:
                return

            visit_SetComp = visit_ListComp
            visit_DictComp = visit_ListComp
            visit_GeneratorExp = visit_ListComp

        def function_arguments(scope: ast.AST) -> set[str]:
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return set()
            arguments = scope.args
            names = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            }
            if arguments.vararg is not None:
                names.add(arguments.vararg.arg)
            if arguments.kwarg is not None:
                names.add(arguments.kwarg.arg)
            return names

        def analyze_scope(scope: ast.AST, enclosing: set[str]) -> None:
            collector = BindingCollector()
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for statement in scope.body:
                    collector.visit(statement)
            else:
                collector.visit(scope)
            local = (collector.bound | function_arguments(scope)) - collector.global_names

            class LoadCollector(ast.NodeVisitor):
                def visit_Name(self, child: ast.Name) -> None:
                    if not isinstance(child.ctx, ast.Load):
                        return
                    if child.id in collector.global_names or (
                        child.id not in local and child.id not in enclosing
                    ):
                        external.add(child.id)

                def _visit_function_header(
                    self,
                    child: ast.FunctionDef | ast.AsyncFunctionDef,
                ) -> None:
                    for decorator in child.decorator_list:
                        self.visit(decorator)
                    for default in (*child.args.defaults, *child.args.kw_defaults):
                        if default is not None:
                            self.visit(default)
                    if child.returns is not None:
                        self.visit(child.returns)
                    analyze_scope(child, enclosing | local)

                def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                    self._visit_function_header(child)

                def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
                    self._visit_function_header(child)

                def visit_ClassDef(self, child: ast.ClassDef) -> None:
                    for decorator in child.decorator_list:
                        self.visit(decorator)
                    for base in child.bases:
                        self.visit(base)
                    for keyword in child.keywords:
                        self.visit(keyword.value)
                    analyze_scope(child, enclosing | local)

                def visit_Lambda(self, child: ast.Lambda) -> None:
                    analyze_scope(child, enclosing | local)

                def _visit_comprehension(
                    self,
                    child: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
                ) -> None:
                    if not child.generators:
                        return
                    previous_local = local.copy()
                    try:
                        for index, generator in enumerate(child.generators):
                            self.visit(generator.iter)
                            local.update(
                                name.id
                                for name in ast.walk(generator.target)
                                if isinstance(name, ast.Name)
                            )
                            for condition in generator.ifs:
                                self.visit(condition)
                        if isinstance(child, ast.DictComp):
                            self.visit(child.key)
                            self.visit(child.value)
                        else:
                            self.visit(child.elt)
                    finally:
                        local.clear()
                        local.update(previous_local)

                visit_ListComp = _visit_comprehension
                visit_SetComp = _visit_comprehension
                visit_DictComp = _visit_comprehension
                visit_GeneratorExp = _visit_comprehension

            loader = LoadCollector()
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in scope.decorator_list:
                        loader.visit(decorator)
                    for default in (*scope.args.defaults, *scope.args.kw_defaults):
                        if default is not None:
                            loader.visit(default)
                    if scope.returns is not None:
                        loader.visit(scope.returns)
                else:
                    for decorator in scope.decorator_list:
                        loader.visit(decorator)
                    for base in scope.bases:
                        loader.visit(base)
                    for keyword in scope.keywords:
                        loader.visit(keyword.value)
                for statement in scope.body:
                    loader.visit(statement)
            elif isinstance(scope, ast.Lambda):
                loader.visit(scope.body)
            else:
                loader.visit(scope)

        analyze_scope(node, set())
        return external - {"self", "cls", "FreshCandidateModel"}

    if added_init_statements and not replace_init_whole:
        if repaired_init is None:
            raise ValueError(
                "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: constructor "
                "dependencies have no constructor scope"
            )
        local_bindings: dict[str, list[tuple[int, ast.stmt]]] = {}
        for index, statement in enumerate(repaired_init.body):
            for name in module_bound_names(statement):
                local_bindings.setdefault(name, []).append((index, statement))
        local_name_universe = set(local_bindings)

        def project_local_binding(statement: ast.stmt, name: str) -> list[ast.stmt]:
            if isinstance(statement, ast.Import):
                return [
                    ast.Import(names=[deepcopy(alias)])
                    for alias in statement.names
                    if (alias.asname or alias.name.split(".", 1)[0]) == name
                ]
            if isinstance(statement, ast.ImportFrom):
                return [
                    ast.ImportFrom(
                        module=statement.module,
                        names=[deepcopy(alias)],
                        level=statement.level,
                    )
                    for alias in statement.names
                    if alias.name != "*" and (alias.asname or alias.name) == name
                ]
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return [deepcopy(statement)] if statement.name == name else []
            return project_binding_statement(statement, {name})

        def loaded_local_names(statement: ast.stmt) -> set[str]:
            # external_name_loads intentionally treats a name assigned by the
            # same statement as local.  Reaching-definition slicing must also
            # see normalization chains such as ``x = x.astype(...)``.
            return {
                child.id
                for child in ast.walk(statement)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            } & local_name_universe

        def nested_control_bindings(statement: ast.stmt) -> set[str]:
            """Names conditionally bound by one top-level constructor row."""

            if not isinstance(
                statement,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                ),
            ):
                return set()

            class ConditionalBindingCollector(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.names: set[str] = set()

                def visit_Name(self, child: ast.Name) -> None:
                    if isinstance(child.ctx, (ast.Store, ast.Del)):
                        self.names.add(child.id)

                def visit_Import(self, child: ast.Import) -> None:
                    self.names.update(
                        alias.asname or alias.name.split(".", 1)[0]
                        for alias in child.names
                    )

                def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
                    self.names.update(
                        alias.asname or alias.name
                        for alias in child.names
                        if alias.name != "*"
                    )

                def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                    self.names.add(child.name)

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_ClassDef(self, child: ast.ClassDef) -> None:
                    self.names.add(child.name)

                def visit_Lambda(self, child: ast.Lambda) -> None:
                    return

            collector = ConditionalBindingCollector()
            collector.visit(statement)
            return collector.names

        def exhaustive_if_binding(statement: ast.stmt, name: str) -> bool:
            """Whether one if row leaves ``name`` bound on every branch."""

            def block_leaves_bound(
                statements: list[ast.stmt],
                initially_bound: bool,
            ) -> bool:
                bound = initially_bound
                for item in statements:
                    if isinstance(item, ast.If):
                        bound = block_leaves_bound(item.body, bound) and (
                            block_leaves_bound(item.orelse, bound)
                            if item.orelse
                            else bound
                        )
                        continue
                    if isinstance(item, ast.AugAssign):
                        # Augmented assignment requires an earlier binding.
                        continue
                    if isinstance(item, ast.AnnAssign) and item.value is None:
                        continue
                    if name in module_bound_names(item):
                        bound = True
                        continue
                    if any(
                        isinstance(child, ast.Name)
                        and child.id == name
                        and isinstance(child.ctx, ast.Del)
                        for child in ast.walk(item)
                    ):
                        bound = False
                return bound

            return (
                isinstance(statement, ast.If)
                and bool(statement.orelse)
                and block_leaves_bound(statement.body, False)
                and block_leaves_bound(statement.orelse, False)
            )

        conditional_bindings = {
            index: nested_control_bindings(statement)
            for index, statement in enumerate(repaired_init.body)
        }
        for names in conditional_bindings.values():
            local_name_universe.update(names)
        selected_init_row_indices = {
            index for index, _statement in added_init_statement_rows
        }
        selected_local_rows: set[tuple[int, str]] = set()
        pending_local_uses = [
            (name, index)
            for index, statement in added_init_statement_rows
            for name in loaded_local_names(statement)
        ]
        while pending_local_uses:
            name, before_index = pending_local_uses.pop()
            candidates = [
                (index, statement)
                for index, statement in local_bindings[name]
                if index < before_index
            ] if name in local_bindings else []
            last_direct_index = candidates[-1][0] if candidates else -1
            exhaustive_control_indices = [
                index
                for index in range(before_index)
                if name in conditional_bindings[index]
                and exhaustive_if_binding(repaired_init.body[index], name)
            ]
            last_control_index = (
                exhaustive_control_indices[-1]
                if exhaustive_control_indices
                else -1
            )
            anchor_index = max(last_direct_index, last_control_index)
            if anchor_index < 0:
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                    f"local binding {name!r} has no reaching definition"
                )
            reaching_control_rows = [
                index
                for index in range(
                    anchor_index if last_control_index > last_direct_index
                    else anchor_index + 1,
                    before_index,
                )
                if index not in selected_init_row_indices
                and name in conditional_bindings[index]
            ]
            for index in reaching_control_rows:
                statement = repaired_init.body[index]
                assigned_self = self_assignments(statement)
                loaded_self = self_loads(statement)
                allowed_self = (
                    required_attributes
                    | available_attributes
                    | set(_SEALED_PARENT_ATTRIBUTES)
                )
                related_cache_mutations = {
                    attribute
                    for attribute in assigned_self - allowed_self
                    if attribute in loaded_self
                    and attribute in repaired_init_statements
                }
                for attribute in related_cache_mutations:
                    initializer = repaired_init_statements[attribute]
                    initializer_index = repaired_init.body.index(initializer)
                    if initializer_index >= index:
                        continue
                    required_attributes.add(attribute)
                    available_attributes.add(attribute)
                    if initializer_index not in selected_init_row_indices:
                        added_init_statement_rows.extend(
                            (initializer_index, item)
                            for item in project_init_statement(initializer)
                        )
                        selected_init_row_indices.add(initializer_index)
                        pending_local_uses.extend(
                            (dependency, initializer_index)
                            for dependency in loaded_local_names(initializer)
                        )
                allowed_self = (
                    required_attributes
                    | available_attributes
                    | set(_SEALED_PARENT_ATTRIBUTES)
                )
                unowned_self = (assigned_self | loaded_self) - allowed_self
                # A pure-local normalization branch is already one atomic source
                # row. Carry it intact; only reject control flow that crosses
                # unrelated candidate state.
                if unowned_self:
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                        f"local binding {name!r} has conditional reaching definitions"
                        + (
                            f" touching {sorted(unowned_self)}"
                            if unowned_self
                            else ""
                        )
                    )
                added_init_statement_rows.append((index, deepcopy(statement)))
                selected_init_row_indices.add(index)
                pending_local_uses.extend(
                    (dependency, index)
                    for dependency in loaded_local_names(statement)
                )
            if last_control_index > last_direct_index:
                continue
            index, statement = candidates[-1]
            row_key = (index, name)
            if row_key in selected_local_rows:
                continue
            projected = project_local_binding(statement, name)
            if not projected:
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: constructor "
                    f"local binding {name!r} cannot be projected"
                )
            added_init_statement_rows.extend((index, item) for item in projected)
            selected_local_rows.add(row_key)
            pending_local_uses.extend(
                (dependency, index)
                for item in projected
                for dependency in loaded_local_names(item)
            )
        added_init_statements = [
            statement
            for _index, statement in sorted(
                added_init_statement_rows,
                key=lambda item: item[0],
            )
        ]
        composed_init = deepcopy(current_init or repaired_init)
        if transactional_init:
            assert current_init is not None
            assert constructor_transaction_current_index is not None
            retained_dumps = {
                ast.dump(statement, include_attributes=False)
                for index, statement in enumerate(current_init.body)
                if index not in constructor_transaction_current_indices
            }
            transaction_rows: list[ast.stmt] = []
            transaction_dumps: set[str] = set()
            for statement in added_init_statements:
                dump = ast.dump(statement, include_attributes=False)
                if dump in retained_dumps or dump in transaction_dumps:
                    continue
                transaction_rows.append(deepcopy(statement))
                transaction_dumps.add(dump)
            composed_init.body = [
                deepcopy(statement)
                for index, current_statement in enumerate(current_init.body)
                for statement in (
                    transaction_rows
                    if index == constructor_transaction_current_index
                    else [] if index in constructor_transaction_current_indices
                    else [current_statement]
                )
            ]
            if not composed_init.body:
                composed_init.body = [ast.Pass()]
        else:
            if current_init is None:
                composed_init.body = []
            insert_at = next(
                (
                    index
                    for index, statement in enumerate(composed_init.body)
                    if isinstance(statement, ast.Return)
                ),
                len(composed_init.body),
            )
            composed_init.body[insert_at:insert_at] = deepcopy(added_init_statements)
        for index, statement in enumerate(current_model.body):
            if (
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == initializer_name
            ):
                current_model.body[index] = composed_init
                break
        else:
            current_model.body.insert(0, composed_init)

    if transactional_init and not added_init_statements:
        assert current_init is not None
        assert constructor_transaction_current_index is not None
        composed_init = deepcopy(current_init)
        composed_init.body = [
            deepcopy(statement)
            for index, statement in enumerate(current_init.body)
            if index not in constructor_transaction_current_indices
        ]
        if not composed_init.body:
            composed_init.body = [ast.Pass()]
        for index, statement in enumerate(current_model.body):
            if (
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == initializer_name
            ):
                current_model.body[index] = composed_init
                break

    if (
        current_init is not None
        and repaired_init is not None
        and initializer_name not in selected
        and not transactional_init
    ):
        # A dependency can exist in both constructors yet be unavailable when
        # an implicated helper first uses it. Preserve a supplied move of the
        # same assignment before that use; do not adopt the whole constructor.
        composed_init = next(
            node for node in current_model.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == initializer_name
        )

        def constructor_reads(statement: ast.stmt) -> set[str]:
            called = method_closure(method_dependencies(statement, repaired_methods))
            return self_loads(statement) | {
                attribute for name in called & selected
                for attribute in self_loads(repaired_methods[name])
            }

        for repaired_index, statement in enumerate(repaired_init.body):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            assigned = self_assignments(statement)
            if not assigned or not assigned <= required_attributes:
                continue
            dump = ast.dump(statement, include_attributes=False)
            if any(
                attribute not in current_init_statements_by_attribute
                or ast.dump(current_init_statements_by_attribute[attribute], include_attributes=False) != dump
                for attribute in assigned
            ):
                continue
            positions = [
                index for index, row in enumerate(composed_init.body)
                if ast.dump(row, include_attributes=False) == dump
            ]
            if len(positions) != 1:
                continue
            current_index = positions[0]
            current_use = next(
                (index for index, row in enumerate(composed_init.body)
                 if constructor_reads(row) & assigned),
                len(composed_init.body),
            )
            repaired_use = next(
                (index for index, row in enumerate(repaired_init.body)
                 if constructor_reads(row) & assigned),
                len(repaired_init.body),
            )
            if current_use < current_index and repaired_index < repaired_use:
                # Keep the reaching bindings of the moved expression. A paid
                # revision may also move/change a local prerequisite that this
                # method-only composition did not retain.
                names = external_name_loads(statement)
                attributes = self_loads(statement)

                def prerequisite_bindings(rows: Sequence[ast.stmt]) -> dict[str, str]:
                    bindings: dict[str, str] = {}
                    for row in rows:
                        keys = (module_bound_names(row) & names) | {
                            f"self.{name}" for name in self_assignments(row) & attributes
                        }
                        for key in keys:
                            bindings[key] = ast.dump(row, include_attributes=False)
                    return bindings

                if prerequisite_bindings(composed_init.body[:current_use]) != prerequisite_bindings(
                    repaired_init.body[:repaired_index]
                ):
                    continue
                composed_init.body.insert(current_use, composed_init.body.pop(current_index))

    dependency_nodes: list[ast.AST] = list(selected_method_nodes)
    dependency_nodes.extend(
        node
        for node in current_model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in consumer_statement_transactions
    )
    if added_init_statements:
        init_dependency_scope = deepcopy(repaired_init)
        init_dependency_scope.body = deepcopy(added_init_statements)
        dependency_nodes.append(init_dependency_scope)
    dependency_nodes.extend(added_class_assignments)

    selected_module_functions = selected & module_function_names
    current_tree.body = [
        deepcopy(repaired_methods[node.name])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected_module_functions else node
        for node in current_tree.body
    ]
    new_functions = [
        deepcopy(node) for node in repaired_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected_module_functions
        and node.name not in current_module_functions
    ]
    insert_at = current_tree.body.index(current_model)
    current_tree.body[insert_at:insert_at] = new_functions

    required_globals = {
        name
        for node in dependency_nodes
        for name in external_name_loads(node)
    }
    duplicate_module_bindings = {
        name
        for name in required_globals
        if current_module_binding_counts.get(name, 0) > 1
        or repaired_module_binding_counts.get(name, 0) > 1
    }
    if duplicate_module_bindings:
        raise ValueError(
            "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required module "
            f"bindings repeat {sorted(duplicate_module_bindings)}"
        )
    added_module_names: dict[int, set[str]] = {}
    pending_globals = list(required_globals)
    unresolved_globals: set[str] = set()
    while pending_globals:
        name = pending_globals.pop()
        if hasattr(builtins, name):
            continue
        if name in selected_module_functions:
            # Its body is already part of dependency_nodes; it was replaced at
            # module scope rather than transplanted into the model class.
            continue
        if name in current_module_bindings:
            repaired_binding = repaired_module_bindings.get(name)
            if (
                repaired_binding is not None
                and projected_module_binding_dump(current_module_bindings[name], name)
                != projected_module_binding_dump(repaired_binding, name)
            ):
                raise ValueError(
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required "
                    f"module binding {name!r} differs"
                )
            continue
        node = repaired_module_bindings.get(name)
        if node is None:
            unresolved_globals.add(name)
            continue
        names = added_module_names.setdefault(id(node), set())
        if name in names:
            continue
        names.add(name)
        pending_globals.extend(
            external_name_loads(node)
        )
    if unresolved_globals:
        raise ValueError(
            "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: repaired symbols load "
            f"undefined globals {sorted(unresolved_globals)}"
        )
    additions: list[ast.stmt] = []
    if added_module_names:
        for node in repaired_tree.body:
            required_names = added_module_names.get(id(node))
            if not required_names:
                continue
            if isinstance(node, ast.Import):
                additions.append(
                    ast.Import(
                        names=[
                            deepcopy(alias)
                            for alias in node.names
                            if (alias.asname or alias.name.split(".", 1)[0])
                            in required_names
                        ]
                    )
                )
            elif isinstance(node, ast.ImportFrom):
                additions.append(
                    ast.ImportFrom(
                        module=node.module,
                        names=[
                            deepcopy(alias)
                            for alias in node.names
                            if (alias.asname or alias.name) in required_names
                        ],
                        level=node.level,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                additions.append(deepcopy(node))
            else:
                additions.extend(project_binding_statement(node, required_names))
        insert_at = current_tree.body.index(current_model)
        current_tree.body[insert_at:insert_at] = additions

    ast.fix_missing_locations(current_tree)
    scoped = dict(current_source)
    scoped[path] = ast.unparse(current_tree) + "\n"

    def local_import_paths(
        node: ast.AST,
        *,
        importer_path: str,
    ) -> set[str]:
        paths: set[str] = set()
        package = importer_path.rsplit("/", 1)[0].split("/")
        for child in ast.walk(node):
            modules: list[str] = []
            if isinstance(child, ast.Import):
                modules.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if any(alias.name == "*" for alias in child.names):
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: wildcard "
                        "candidate-local import cannot be composed"
                    )
                if child.level:
                    keep = max(0, len(package) - (child.level - 1))
                    prefix = package[:keep]
                    if child.module:
                        prefix.extend(child.module.split("."))
                    base = ".".join(prefix)
                    if child.module:
                        modules.append(base)
                    else:
                        modules.extend(
                            f"{base}.{alias.name}" for alias in child.names
                        )
                elif child.module:
                    modules.append(child.module)
                    if child.module == "recclaw_ext":
                        modules.extend(
                            f"recclaw_ext.{alias.name}" for alias in child.names
                        )
            for module in modules:
                if not module.startswith("recclaw_ext."):
                    continue
                dependency_path = module.replace(".", "/") + ".py"
                if (
                    dependency_path in current_source
                    or dependency_path in repaired_source
                ):
                    paths.add(dependency_path)
                elif module != "recclaw_ext":
                    raise ValueError(
                        "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: "
                        f"candidate-local import {module!r} has no source file"
                    )
        return paths

    import_dependency_nodes: list[ast.AST] = [
        *selected_method_nodes,
        *added_init_statements,
        *added_class_assignments,
        *additions,
    ]
    for name in required_globals:
        binding = repaired_module_bindings.get(name) or current_module_bindings.get(name)
        if isinstance(binding, (ast.Import, ast.ImportFrom)):
            import_dependency_nodes.append(binding)
    pending_local_files = [
        dependency_path
        for node in import_dependency_nodes
        for dependency_path in local_import_paths(node, importer_path=path)
    ]
    visited_local_files: set[str] = set()
    while pending_local_files:
        dependency_path = pending_local_files.pop()
        if dependency_path in visited_local_files:
            continue
        visited_local_files.add(dependency_path)
        current_dependency = current_source.get(dependency_path)
        repaired_dependency = repaired_source.get(dependency_path)
        if isinstance(current_dependency, str):
            if (
                isinstance(repaired_dependency, str)
                and repaired_dependency != current_dependency
            ):
                try:
                    ast_equivalent = ast.dump(
                        ast.parse(current_dependency, filename=dependency_path),
                        include_attributes=False,
                    ) == ast.dump(
                        ast.parse(repaired_dependency, filename=dependency_path),
                        include_attributes=False,
                    )
                except SyntaxError as error:
                    raise ValueError(
                        "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: candidate-local "
                        f"file {dependency_path!r} is not valid Python"
                    ) from error
                if ast_equivalent:
                    dependency_source = current_dependency
                elif dependency_path in implicated_files:
                    dependency_source = repaired_dependency
                    scoped[dependency_path] = dependency_source
                else:
                    raise ValueError(
                        "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS: required "
                        f"candidate-local file {dependency_path!r} differs"
                    )
            else:
                dependency_source = current_dependency
        elif isinstance(repaired_dependency, str):
            dependency_source = repaired_dependency
            scoped[dependency_path] = dependency_source
        else:
            raise ValueError(
                "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: candidate-local "
                f"file {dependency_path!r} is absent"
            )
        try:
            dependency_tree = ast.parse(
                dependency_source,
                filename=dependency_path,
            )
        except SyntaxError as error:
            raise ValueError(
                "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED: candidate-local "
                f"file {dependency_path!r} is not valid Python"
            ) from error
        pending_local_files.extend(
            local_import_paths(dependency_tree, importer_path=dependency_path)
            - visited_local_files
        )
    return scoped


def validate_mechanical_repair_source_scope(
    *,
    current_source: Mapping[str, str],
    repaired_source: Mapping[str, str],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject mechanical repairs that rewrite an unrelated scoring interface.

    A repair is allowed to change the scorer method named by the mechanical
    failure. Otherwise ``predict`` and ``full_sort_predict`` must remain
    semantically identical at the Python AST boundary. This keeps a loss,
    construction, or resource repair from silently changing the compiled
    BL-ICF score-head mechanism.
    """

    current = current_source.get("recclaw_ext/candidate.py")
    repaired = repaired_source.get("recclaw_ext/candidate.py")
    if not isinstance(current, str) or not isinstance(repaired, str):
        raise ValueError("mechanical repair must preserve recclaw_ext/candidate.py")
    try:
        before = _class_methods(current)
        after = _class_methods(repaired)
    except SyntaxError as error:
        raise ValueError("mechanical repair scorer scope is not valid Python") from error
    implicated_methods = _implicated_methods(failure)
    evidence = "\n".join(
        str(failure.get(field, "")).lower()
        for field in (
            "stage",
            "failure_class",
            "reason_code",
            "message",
            "short_trace",
            "traceback",
        )
    )
    preserved: list[str] = []
    allowed: list[str] = []

    for method in _SCORER_METHODS:
        if (
            method in implicated_methods
            if implicated_methods
            else method.lower() in evidence
        ):
            allowed.append(method)
            continue
        if _method_fingerprint(before, method) != _method_fingerprint(after, method):
            raise ValueError(
                f"mechanical repair changed failure-unrelated scorer method {method}"
            )
        preserved.append(method)
    return canonical_value(
        {
            "allowed_scorer_methods": allowed,
            "preserved_scorer_methods": preserved,
            "schema": "recclaw.mechanical-repair-source-scope.v1",
        }
    )


def build_conversion_execution_plan(
    candidate_ids: Sequence[str],
    *,
    screen_seed: int,
    full_seeds: Sequence[int] = FULL_DEVELOPMENT_SEEDS,
    promotion_limit: int = 2,
) -> dict[str, Any]:
    """Freeze the short-fidelity and fresh-seed full-run schedule."""

    ids = tuple(str(value) for value in candidate_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("conversion plan requires unique candidate identities")
    seeds = tuple(int(value) for value in full_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or int(screen_seed) in seeds:
        raise ValueError("full seeds must be distinct and fresh from the screen seed")
    if int(promotion_limit) < 1 or int(promotion_limit) > len(ids):
        raise ValueError("promotion limit is outside the candidate denominator")
    payload = canonical_value(
        {
            "schema": CONVERSION_SCHEMA,
            "candidate_ids": ids,
            "screen": {
                "epochs": SCREEN_EPOCHS,
                "seed": int(screen_seed),
                "one_pristine_parent_per_seed": True,
                "failure_is_missing": True,
                "deadline_rule": {
                    "source": "ONE_EPOCH_SMOKE_QUALIFICATION_TELEMETRY",
                    "startup_margin_seconds": SCREEN_DEADLINE_STARTUP_MARGIN_SECONDS,
                    "variance_margin": SCREEN_DEADLINE_VARIANCE_MARGIN,
                    "formula": "max(180, ceil(60 + 1.25 * one_epoch_smoke_seconds * 20))",
                    "watchdog_seconds": FULL_WATCHDOG_SECONDS,
                },
            },
            "promotion": {
                "limit": int(promotion_limit),
                "requires_completed_stable_screen": True,
                "selection_input": "SCREEN_STATUS_AND_STABILITY_ONLY",
            },
            "full": {
                "epochs": FULL_EPOCHS,
                "fresh_development_seeds": seeds,
                "one_pristine_parent_per_seed": True,
                "screen_outcomes_reused_as_effect": False,
            },
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "plan_digest": sha256_digest(payload)}


def derive_resource_deadline_seconds(
    screen_wall_time_ms: int | float,
    *,
    screen_epochs: int = SCREEN_EPOCHS,
    full_epochs: int = FULL_EPOCHS,
) -> int:
    """Project a full-run deadline from pre-outcome screen wall time."""

    wall_seconds = float(screen_wall_time_ms) / 1000.0
    if wall_seconds <= 0 or int(screen_epochs) <= 0 or int(full_epochs) <= 0:
        raise ValueError("screen telemetry and epoch counts must be positive")
    projected = ceil(
        60.0
        + RESOURCE_DEADLINE_MARGIN
        * (wall_seconds * int(full_epochs) / int(screen_epochs))
    )
    deadline = max(RESOURCE_DEADLINE_MIN_SECONDS, projected)
    if deadline > FULL_WATCHDOG_SECONDS:
        raise ValueError("projected deadline exceeds the frozen watchdog")
    return int(deadline)


def derive_screen_deadline_seconds(
    one_epoch_smoke_wall_time_ms: int | float,
    *,
    smoke_epochs: int = 1,
    screen_epochs: int = SCREEN_EPOCHS,
) -> int:
    """Freeze a screen deadline from pre-screen one-epoch smoke telemetry."""

    wall_seconds = float(one_epoch_smoke_wall_time_ms) / 1000.0
    if wall_seconds <= 0 or int(smoke_epochs) <= 0 or int(screen_epochs) <= 0:
        raise ValueError("one-epoch smoke telemetry and epoch counts must be positive")
    projected = ceil(
        SCREEN_DEADLINE_STARTUP_MARGIN_SECONDS
        + SCREEN_DEADLINE_VARIANCE_MARGIN
        * (wall_seconds * int(screen_epochs) / int(smoke_epochs))
    )
    deadline = max(RESOURCE_DEADLINE_MIN_SECONDS, projected)
    if deadline > FULL_WATCHDOG_SECONDS:
        raise ValueError("screen deadline exceeds the frozen watchdog")
    return int(deadline)


def choose_stable_promotions(
    screen_results: Sequence[Mapping[str, Any]],
    *,
    promotion_limit: int,
) -> tuple[str, ...]:
    """Choose a fixed small full-run set from completed screen telemetry only."""

    eligible = []
    for row in screen_results:
        if (
            row.get("status") == "COMPLETED_MATCHED_SCREEN"
            and bool(row.get("stable"))
            and isinstance(row.get("candidate_id"), str)
        ):
            eligible.append(row)
    ordered = sorted(
        eligible,
        key=lambda row: _screen_priority_key(
            row,
            predicted_cost_seconds=_predicted_full_cost_seconds(row),
        ),
    )
    return tuple(str(row["candidate_id"]) for row in ordered[: int(promotion_limit)])


def _predicted_full_cost_seconds(
    row: Mapping[str, Any],
    *,
    seed_count: int = len(FULL_DEVELOPMENT_SEEDS),
) -> int:
    wall_time_ms = row.get("screen_cost_ms")
    if wall_time_ms is None:
        return FULL_WATCHDOG_SECONDS
    try:
        return derive_resource_deadline_seconds(wall_time_ms) * int(seed_count)
    except ValueError:
        return FULL_WATCHDOG_SECONDS


def _screen_priority_key(
    row: Mapping[str, Any],
    *,
    predicted_cost_seconds: int,
) -> tuple[int, float, float, int, str]:
    """Prefer positive signal, then signal per predicted full-run cost."""

    signal = float(row.get("screen_signal", 0.0) or 0.0)
    positive_rank = 0 if signal > 0 else 1
    efficiency = signal / max(1, int(predicted_cost_seconds))
    return (
        positive_rank,
        -efficiency,
        -signal,
        int(predicted_cost_seconds),
        str(row["candidate_id"]),
    )


def choose_resource_bounded_promotions(
    screen_results: Sequence[Mapping[str, Any]],
    *,
    promotion_limit: int,
    full_seeds: Sequence[int] = FULL_DEVELOPMENT_SEEDS,
    total_budget_seconds: int = FULL_RESOURCE_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Apply the fixed screen/policy order to the full-run resource budget."""

    seeds = tuple(int(seed) for seed in full_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("full seeds must be unique")
    parent_wall_time_ms = next(
        (
            row.get("parent_screen_cost_ms")
            for row in screen_results
            if row.get("parent_screen_cost_ms") is not None
        ),
        None,
    )
    if parent_wall_time_ms is None:
        raise ValueError("stable screen results require shared-parent wall time")
    parent_deadline = derive_resource_deadline_seconds(parent_wall_time_ms)
    parent_reserve = parent_deadline * len(seeds)
    used = parent_reserve
    eligible_rows = [
        row
        for row in screen_results
        if row.get("status") == "COMPLETED_MATCHED_SCREEN"
        and bool(row.get("stable"))
        and isinstance(row.get("candidate_id"), str)
    ]
    ordered_rows = sorted(
        eligible_rows,
        key=lambda row: _screen_priority_key(
            row,
            predicted_cost_seconds=_predicted_full_cost_seconds(
                row, seed_count=len(seeds)
            ),
        ),
    )
    ordered = tuple(
        str(row["candidate_id"])
        for row in ordered_rows[: int(promotion_limit)]
    )
    promoted: list[str] = []
    censored: list[dict[str, Any]] = []
    deadlines: dict[str, int] = {}
    candidate_costs: dict[str, int] = {}
    by_id = {str(row["candidate_id"]): row for row in screen_results}
    for candidate_id in ordered:
        wall_time_ms = by_id[candidate_id].get("screen_cost_ms")
        if wall_time_ms is None:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "SCREEN_WALL_TIME_MISSING",
                }
            )
            continue
        try:
            deadline = derive_resource_deadline_seconds(wall_time_ms)
        except ValueError:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "DEADLINE_EXCEEDS_WATCHDOG",
                }
            )
            continue
        cost = deadline * len(seeds)
        deadlines[candidate_id] = deadline
        candidate_costs[candidate_id] = cost
        if used + cost <= int(total_budget_seconds):
            promoted.append(candidate_id)
            used += cost
        else:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "FULL_RESOURCE_BUDGET_EXCEEDED",
                    "required_seconds": cost,
                    "remaining_seconds": max(0, int(total_budget_seconds) - used),
                }
            )
    return canonical_value(
        {
            "promoted_candidate_ids": promoted,
            "resource_censored_not_promoted": censored,
            "full_deadline_seconds_by_candidate": deadlines,
            "shared_parent_deadline_seconds": parent_deadline,
            "shared_parent_reserve_seconds": parent_reserve,
            "candidate_full_cost_seconds": candidate_costs,
            "resource_budget_seconds": int(total_budget_seconds),
            "resource_budget_used_seconds": used,
            "screen_priority_order": list(ordered),
            "selection_conditioning": (
                "COMPLETED_STABLE_SCREEN_SIGNAL_THEN_PREDICTED_FULL_COST"
            ),
            "negative_or_exploration_fallback": (
                "BEST_SCREEN_SIGNAL_WHEN_NO_POSITIVE_FEASIBLE_CANDIDATE"
            ),
        }
    )


def finalize_conversion_execution_plan(
    plan: Mapping[str, Any],
    screen_results: Sequence[Mapping[str, Any]],
    promoted_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach the fixed promotions and expose parent/candidate run counts."""

    candidate_ids = {str(value) for value in plan["candidate_ids"]}
    promoted = tuple(str(value) for value in promoted_ids)
    if len(set(promoted)) != len(promoted) or not set(promoted).issubset(candidate_ids):
        raise ValueError("promotion set is not a subset of the frozen candidate denominator")
    screen_seed = int(plan["screen"]["seed"])
    full_seeds = tuple(int(value) for value in plan["full"]["fresh_development_seeds"])
    resource = choose_resource_bounded_promotions(
        screen_results,
        promotion_limit=int(plan["promotion"]["limit"]),
        full_seeds=full_seeds,
    )
    if tuple(promoted) != tuple(resource["promoted_candidate_ids"]):
        raise ValueError("promotion set does not match the frozen resource screen rule")
    payload = canonical_value(
        {
            **dict(plan),
            "promotion": {
                **dict(plan["promotion"]),
                "promoted_candidate_ids": promoted,
                "resource_censored_not_promoted": resource[
                    "resource_censored_not_promoted"
                ],
                "full_deadline_seconds_by_candidate": resource[
                    "full_deadline_seconds_by_candidate"
                ],
                "shared_parent_deadline_seconds": resource[
                    "shared_parent_deadline_seconds"
                ],
                "shared_parent_reserve_seconds": resource[
                    "shared_parent_reserve_seconds"
                ],
                "candidate_full_cost_seconds": resource[
                    "candidate_full_cost_seconds"
                ],
                "selection_conditioning": resource["selection_conditioning"],
                "negative_or_exploration_fallback": resource[
                    "negative_or_exploration_fallback"
                ],
                "resource_budget_seconds": resource["resource_budget_seconds"],
                "resource_budget_used_seconds": resource[
                    "resource_budget_used_seconds"
                ],
            },
            "run_counts": {
                "screen_candidate_runs": len(candidate_ids),
                "screen_parent_runs": 1,
                "full_candidate_runs": len(promoted) * len(full_seeds),
                "full_parent_runs": len(full_seeds),
                "shared_parent_runs": 1 + len(full_seeds),
                "screen_seed": screen_seed,
                "full_seeds": full_seeds,
            },
        }
    )
    return {**payload, "plan_digest": sha256_digest(payload)}


def run_fail_soft_batch(
    candidate_ids: Sequence[str],
    run_one: Any,
) -> tuple[dict[str, Any], ...]:
    """Run each arm independently; one local failure becomes missing."""

    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        normalized = str(candidate_id)
        try:
            result = run_one(normalized)
        except Exception as error:  # noqa: BLE001 - failure isolation is the contract.
            rows.append(
                {
                    "candidate_id": normalized,
                    "status": "MISSING",
                    "failure_class": type(error).__name__,
                }
            )
        else:
            rows.append(
                {
                    "candidate_id": normalized,
                    "status": "COMPLETED",
                    "result": result,
                }
            )
    return tuple(rows)


def build_stage_feasibility_head(
    stage_labels: Sequence[Mapping[str, Any]],
    *,
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Update only feasibility from the 17-arm stage labels.

    Effect remains strongly shrunk because Q5-A supplied only two complete
    Episodes; this function never consumes effect values or selects ideas.
    """

    labels = tuple(stage_labels)
    denominator = len(labels)
    if denominator != 17:
        raise ValueError("Q5 conversion feasibility head requires the frozen 17-arm labels")
    stages = ("CONSTRUCT", "MATERIALIZE", "QUALIFY", "RESOURCE_ADMITTED", "FULL_EPISODE")
    stage_rates = {
        stage: sum(row.get(stage) == "PASS" for row in labels) / denominator
        for stage in stages
    }
    by_policy: dict[str, dict[str, float]] = {}
    for policy in policy_order:
        rows = [row for row in labels if policy in tuple(row.get("policies", ()))]
        count = len(rows)
        by_policy[str(policy)] = {
            stage: (sum(row.get(stage) == "PASS" for row in rows) / count if count else 0.0)
            for stage in stages
        }
    values = tuple(
        round(by_policy[str(policy)]["RESOURCE_ADMITTED"], 12)
        for policy in policy_order
    )
    feature_names = sorted(
        {
            str(name)
            for row in labels
            for name in (
                row.get("preoutcome_features", {})
                if isinstance(row.get("preoutcome_features", {}), Mapping)
                else {}
            )
        }
    )
    feature_calibration: dict[str, Any] = {}
    for feature in feature_names:
        observed = []
        for row in labels:
            features = row.get("preoutcome_features", {})
            if not isinstance(features, Mapping):
                continue
            value = features.get(feature)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                observed.append((float(value), row))
        if not observed:
            continue
        bands = {
            "LOW": [row for value, row in observed if value < 0.5],
            "HIGH": [row for value, row in observed if value >= 0.5],
        }
        feature_calibration[feature] = {
            "observed_count": len(observed),
            "authority": "PRE_OUTCOME_STRUCTURAL_FEATURE_ONLY",
            "bands": {
                band: {
                    "count": len(rows),
                    "stage_completion": {
                        stage: (
                            sum(row.get(stage) == "PASS" for row in rows) / len(rows)
                            if rows
                            else None
                        )
                        for stage in stages
                    },
                }
                for band, rows in bands.items()
            },
        }
    return canonical_value(
        {
            "schema": CONVERSION_SCHEMA,
            "stage_denominator": denominator,
            "stage_completion_head": stage_rates,
            "policy_stage_completion_head": by_policy,
            "policy_output": "NONUNIFORM" if len(set(values)) > 1 else "TIE",
            "preoutcome_feature_calibration": feature_calibration,
            "effect_head": {
                "observed_full_episode_count": sum(row.get("FULL_EPISODE") == "PASS" for row in labels),
                "shrinkage": "STRONG",
                "claim_allowed": False,
            },
            "held_out_reads": 0,
            "outcome_leakage": False,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )


__all__ = [
    "CANDIDATE_LOCAL_ALLOWED_FILES",
    "CONVERSION_SCHEMA",
    "FULL_DEVELOPMENT_SEEDS",
    "FULL_EPOCHS",
    "FULL_RESOURCE_BUDGET_SECONDS",
    "FULL_WATCHDOG_SECONDS",
    "MAX_REPAIR_TURNS",
    "MECHANICAL_REPAIR_SCHEMA",
    "MECHANICAL_SHAPE_TEST_HINT",
    "RECBole_INTERFACE_CONTRACT",
    "SCREEN_EPOCHS",
    "build_conversion_execution_plan",
    "build_mechanical_repair_request",
    "build_stage_feasibility_head",
    "choose_resource_bounded_promotions",
    "choose_stable_promotions",
    "derive_resource_deadline_seconds",
    "derive_screen_deadline_seconds",
    "finalize_conversion_execution_plan",
    "is_mechanical_repair_failure",
    "normalize_qualification_failure",
    "run_fail_soft_batch",
    "validate_mechanical_repair_source_scope",
]
