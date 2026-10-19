from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import hashlib
import ast
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

from gagc.grpo import (
    COMPOSITE_ARMS,
    GR_COMPOSITE_ARMS,
    GR_JUMPING_DIMS,
    GR_MUTEX_GROUPS,
    JUMPING_DIMS,
    MUTEX_GROUPS,
    compute_group_advantages,
    thompson_sample,
)
from gagc.schemas import MutationSpec, TrialResult
from gagc.state import ArmState, ThompsonState

# ---------------------------------------------------------------------------
# Penalty scores for failed runs
# ---------------------------------------------------------------------------
_TIMEOUT_SCORE = -1.0
_OOM_SCORE = -0.5
_CRASH_SCORE = 0.0

# Module-level store reference — injected by agent.py so update_thompson_state
# can persist state directly without LLM write_file involvement.
_STORE: Any = None
_STORE_NAMESPACE: tuple = ("gagc", "thompson")
_STORE_KEY: str = "/state.json"
_LOGS_ROOT: str = ""
_LAST_SELECTION_LOG: dict[str, Any] = {}
_LAST_PROPOSED_CANDIDATES: list[dict[str, Any]] = []
_LAST_TRIAL_RESULTS: list[dict[str, Any]] = []
_LAST_TRIAL_GROUP_ID: str = ""
_LAST_PROMOTION_RESULT: dict[str, Any] = {}
_WORKSPACE_ROOT: str = ""

# Routing policy switch. Keep Thompson enabled by default for existing runs;
# set to "textual_gradient" to run the LLM-memory-only ablation where the
# orchestrator chooses arms from the textual experiment memory.
_ROUTING_POLICY: str = "thompson"

# ---------------------------------------------------------------------------
# Benchmark mode switch
# ---------------------------------------------------------------------------
_BENCHMARK_MODE: str = "amazon_reviews"
_GR_TRAIN_DATA: str = ""
_GR_VAL_DATA: str = ""
_GR_TEST_DATA: str = ""
_E1_DATA_ROOT: str = ""
_E1_SEARCH_ROOT: str = ""
_E1_MANIFEST: str = ""
_E1_MANIFEST_SHA256: str = ""
_E1_DEVICE: str = "auto"
_E1_RUNTIME: Any = None
_E2_RUNTIME: Any = None
_SELECTION_METRIC: str = "val_score"
_STRICT_VALIDATION_ONLY: bool = False

# ---------------------------------------------------------------------------
# Basin parameters
# ---------------------------------------------------------------------------
THETA: float = 0.03       # ceiling gap threshold to open basin-jumping
N_RETUNE: int = 4         # rounds to wait before back-filling jumping dim α/β
MIN_STAGNANT_ROUNDS_BEFORE_JUMP: int = 3  # require consecutive best-score stalls before jumping
BASIN_TRANSFER_RHO: float = 0.25  # one-time posterior transfer when a new basin is accepted


def _incumbent_score_from_state() -> float | None:
    """Return the best persisted validation score for incumbent promotion gates."""
    raw = ""
    if _STORE is not None:
        try:
            item = _STORE.get(_STORE_NAMESPACE, _STORE_KEY)
            raw = item.value["content"] if item else ""
        except Exception:
            raw = ""
    if not raw:
        return None
    try:
        state = ThompsonState.from_json(raw)
    except Exception:
        return None
    return _previous_best_test(state)

# ---------------------------------------------------------------------------
# Server hardware configuration
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Hardware topology for the execution host."""
    num_gpus: int = 8
    num_cpus: int = 128
    cpus_per_trial: int = 16
    gpu_ids: list[int] = field(default_factory=list)
    numa_map: dict[int, list[int]] = field(default_factory=lambda: {
        0: list(range(0, 32)) + list(range(64, 96)),
        1: list(range(32, 64)) + list(range(96, 128)),
    })
    trial_group_execution: str = "parallel"
    logical_group_size: int | None = None

    def cpu_set_for_slot(self, slot_id: int) -> list[int]:
        slots_per_node = max(self.num_gpus // 2, 1)
        node = 0 if slot_id < max(self.num_gpus // 2, 1) else 1
        node_cpus = self.numa_map[node]
        slot_within_node = slot_id % slots_per_node
        start = slot_within_node * self.cpus_per_trial
        return node_cpus[start: start + self.cpus_per_trial]

    def gpu_id_for_slot(self, slot_id: int) -> int:
        if self.gpu_ids:
            return self.gpu_ids[slot_id % len(self.gpu_ids)]
        return slot_id % self.num_gpus


_SERVER_CONFIG: ServerConfig = ServerConfig()


def _execute_trial_members(
    specs: list[dict[str, Any]],
    run_one: Any,
) -> tuple[list[str | None], list[dict[str, str] | None]]:
    """Execute a frozen logical trial group without changing its membership.

    ``serial`` is an execution-topology choice for one-GPU hosts. It does not
    turn a normal G-member Thompson group into G singleton rounds: every member
    is still returned together for one promotion and one posterior update.
    """
    results: list[str | None] = [None] * len(specs)
    mutated_payloads: list[dict[str, str] | None] = [None] * len(specs)
    mode = _SERVER_CONFIG.trial_group_execution.strip().lower()

    if mode == "serial":
        completed = (run_one(index, spec) for index, spec in enumerate(specs))
        for index, result_json, mutated_payload in completed:
            results[index] = result_json
            mutated_payloads[index] = mutated_payload
        return results, mutated_payloads

    if mode != "parallel":
        raise RuntimeError(
            "trial_group_execution must be 'parallel' or 'serial'; "
            f"received {mode!r}"
        )

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(run_one, index, spec): index for index, spec in enumerate(specs)}
        for future in as_completed(futures):
            index, result_json, mutated_payload = future.result()
            results[index] = result_json
            mutated_payloads[index] = mutated_payload
    return results, mutated_payloads


# ---------------------------------------------------------------------------
# Runtime logging helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion for JSON logs."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _configured_basin_transfer_rho() -> float:
    raw = os.getenv("GAGC_BASIN_TRANSFER_RHO", "").strip()
    if not raw:
        return BASIN_TRANSFER_RHO
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return BASIN_TRANSFER_RHO


def _clone_arms(arms: dict[str, ArmState]) -> dict[str, ArmState]:
    return {
        name: ArmState(name=arm.name or name, alpha=float(arm.alpha), beta=float(arm.beta))
        for name, arm in arms.items()
    }


def _ensure_active_arm_entries(state: ThompsonState) -> None:
    for arm in _active_arms():
        state.get_or_create_arm(arm)


def _sync_current_basin(state: ThompsonState) -> None:
    basin_id = getattr(state, "current_basin_id", "") or "default"
    state.current_basin_id = basin_id
    if not getattr(state, "basin_arms", None):
        state.basin_arms = {basin_id: _clone_arms(state.arms)}
    elif basin_id in state.basin_arms:
        state.arms = _clone_arms(state.basin_arms[basin_id])
    else:
        state.basin_arms[basin_id] = _clone_arms(state.arms)
    _ensure_active_arm_entries(state)
    state.basin_arms[basin_id] = _clone_arms(state.arms)


def _new_basin_id(state: ThompsonState, jumping_arm: str) -> str:
    base = getattr(state, "current_basin_id", "") or "default"
    return f"{base}|{jumping_arm}@{state.round_idx}"


def _switch_to_new_basin(state: ThompsonState, jumping_arm: str) -> dict[str, Any]:
    rho = _configured_basin_transfer_rho()
    previous_basin_id = getattr(state, "current_basin_id", "") or "default"
    parent_arms = _clone_arms(state.arms)
    state.basin_arms[previous_basin_id] = parent_arms
    next_basin_id = _new_basin_id(state, jumping_arm)

    if next_basin_id not in state.basin_arms:
        jumping_dims = _active_jumping_dims()
        new_arms: dict[str, ArmState] = {}
        for name, arm in parent_arms.items():
            if name in jumping_dims:
                new_arms[name] = ArmState(
                    name=arm.name or name,
                    alpha=float(arm.alpha),
                    beta=float(arm.beta),
                )
            else:
                new_arms[name] = ArmState(
                    name=arm.name or name,
                    alpha=1.0 + rho * (float(arm.alpha) - 1.0),
                    beta=1.0 + rho * (float(arm.beta) - 1.0),
                )
        state.basin_arms[next_basin_id] = new_arms

    state.current_basin_id = next_basin_id
    state.arms = _clone_arms(state.basin_arms[next_basin_id])
    _ensure_active_arm_entries(state)
    state.basin_arms[next_basin_id] = _clone_arms(state.arms)
    return {
        "previous_basin_id": previous_basin_id,
        "new_basin_id": next_basin_id,
        "rho": rho,
    }


def _switch_to_existing_basin(state: ThompsonState, basin_id: str) -> None:
    state.basin_arms[state.current_basin_id] = _clone_arms(state.arms)
    if basin_id in state.basin_arms:
        state.current_basin_id = basin_id
        state.arms = _clone_arms(state.basin_arms[basin_id])
        _ensure_active_arm_entries(state)
        state.basin_arms[basin_id] = _clone_arms(state.arms)


def _state_summary(state: ThompsonState) -> dict[str, Any]:
    """Compact state snapshot for logs."""
    return {
        "round_idx": state.round_idx,
        "global_budget": state.global_budget,
        "current_basin_id": getattr(state, "current_basin_id", "default"),
        "basin_ids": sorted((getattr(state, "basin_arms", {}) or {}).keys()),
        "arms": {
            name: {"alpha": arm.alpha, "beta": arm.beta}
            for name, arm in state.arms.items()
        },
        "score_history": list(state.score_history),
        "pending_queue": list(state.pending_queue),
        "provisional_incumbent": {
            "active": bool(getattr(state, "provisional_incumbent", {})),
            "score": (getattr(state, "provisional_incumbent", {}) or {}).get("score"),
            "dim": (getattr(state, "provisional_incumbent", {}) or {}).get("dim"),
            "t_start": (getattr(state, "provisional_incumbent", {}) or {}).get("t_start"),
        },
        "recent_text_gradients": list(state.recent_text_gradients),
        "failure_memory": list(state.failure_memory),
        "experiment_skill_hash": _hash_text(state.experiment_skill or ""),
        "experiment_skill": state.experiment_skill,
        "skill_notes": state.skill_notes,
    }


def _write_json_log(relative_path: str, payload: dict[str, Any]) -> None:
    """Write a runtime JSON log under logs_root when configured."""
    if not _LOGS_ROOT:
        return
    try:
        path = os.path.join(_LOGS_ROOT, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    except Exception:
        # Logging must never break optimisation.
        pass


def _spec_to_dict(spec: MutationSpec) -> dict[str, Any]:
    """Return a JSON-safe full spec dict, preserving proposal metadata."""
    if hasattr(spec, "to_dict"):
        return _json_safe(spec.to_dict())
    return {
        "dimension": spec.dimension,
        "delta": spec.delta,
        "estimated_cost_secs": spec.estimated_cost_secs,
        "code_diff": spec.code_diff,
        "code_edits": getattr(spec, "code_edits", []),
        "code_content": getattr(spec, "code_content", ""),
    }


def _resolve_trial_files(script_path: str) -> tuple[str, str]:
    """Return the training entrypoint and predict.py path for a trial workspace."""
    script_path = _resolve_script_path(script_path)
    trial_dir = os.path.dirname(script_path)
    return script_path, os.path.join(trial_dir, "predict.py")


def _resolve_script_path(script_path: str) -> str:
    """Map virtual /workspace paths to the real workspace path for tool execution."""
    raw_path = str(script_path or "").strip() or "/workspace/best.py"
    if raw_path.startswith("/workspace/") and _WORKSPACE_ROOT:
        rel_path = raw_path[len("/workspace/"):].lstrip("/")
        return os.path.join(_WORKSPACE_ROOT, rel_path)
    return raw_path


def _is_canonical_workspace_script(script_path: str) -> bool:
    """Detect direct calls against the incumbent script rather than an isolated slot."""
    return os.path.basename(script_path) == "best.py" and not os.path.basename(os.path.dirname(script_path)).startswith("_trial_")


def _spec_has_mutation(spec: MutationSpec) -> bool:
    return bool(spec.code_edits or spec.code_content.strip() or spec.code_diff.strip())


def _is_baseline_spec_data(spec_data: dict[str, Any]) -> bool:
    return bool(
        spec_data.get("is_baseline")
        or (
            not spec_data.get("code_edits")
            and not spec_data.get("code_content")
            and not spec_data.get("code_diff")
            and float(spec_data.get("delta", 0.0) or 0.0) == 0.0
            and str(spec_data.get("dimension", "")).lower() in {"baseline", "noop", "no_op"}
        )
    )


def _apply_text_mutation(path: str, edits: list[dict], content: str, label: str) -> str | None:
    """Apply find/replace edits or full content to one file; return error text on failure."""
    if edits:
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
            mutated, err = _apply_edits(existing, edits)
            if err:
                return f"{label} edit failed: {err}"
            with open(path, "w", encoding="utf-8") as f:
                f.write(mutated)
        except OSError as exc:
            return f"{label} edit IO error: {exc}"
    elif content.strip():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            return f"{label} write IO error: {exc}"
    return None


def _syntax_check_python(path: str, label: str) -> str | None:
    """Return a concise syntax error for a mutated Python file, if any."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, path, "exec")
    except SyntaxError as exc:
        return f"{label} syntax error at line {exc.lineno}: {exc.msg}"
    except OSError as exc:
        return f"{label} syntax check IO error: {exc}"
    return None


def _python_defined_names(path: str) -> tuple[set[str], str | None]:
    """Return top-level function/class names, or an error if the file cannot parse."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except SyntaxError as exc:
        return set(), f"syntax error at line {exc.lineno}: {exc.msg}"
    except OSError as exc:
        return set(), f"IO error: {exc}"
    names = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return names, None


def _contract_check_trial_files(train_path: str, predict_path: str) -> str | None:
    """Catch common destructive edits before spending GPU time."""
    train_names, err = _python_defined_names(train_path)
    if err:
        label = "E1 candidate" if _BENCHMARK_MODE == "e1_ml1m" else "train.py"
        return f"{label} contract check failed: {err}"
    if _BENCHMARK_MODE == "e1_ml1m":
        if "FreshCandidateModel" not in train_names:
            return "E1 candidate contract check failed: define RecBole FreshCandidateModel"
        return None
    # Training scripts are executed as standalone programs; benchmark evaluation
    # is performed by the RecHarness harness after training, not by functions embedded
    # in the template. Do not require template-local evaluate_val_fast/evaluate.
    required_train_names = {"main"}
    missing_train = sorted(required_train_names - train_names)
    if missing_train:
        return f"train.py contract check failed: missing top-level definitions {missing_train}"

    if _BENCHMARK_MODE == "amazon_reviews" and os.path.exists(predict_path):
        predict_names, err = _python_defined_names(predict_path)
        if err:
            return f"predict.py contract check failed: {err}"
        if "predict" not in predict_names:
            return "predict.py contract check failed: missing top-level predict()"
    return None


def _trial_runtime_env(slot_id: int) -> tuple[dict[str, str], list[int]]:
    """Return the execution environment and CPU affinity for one trial slot."""
    cfg = _SERVER_CONFIG
    cpu_set = cfg.cpu_set_for_slot(slot_id)

    env = os.environ.copy()
    # Claude Code's auto-updater rewrites bin/claude.exe in place (its install.cjs
    # copies the native binary over the placeholder). Under parallel trial load a
    # concurrent spawn can execve the half-written file -> ENOEXEC (errno 8).
    # Trials must be reproducible, so never let a trial self-update mid-run;
    # update the CLI intentionally between runs instead.
    env["DISABLE_AUTOUPDATER"] = "1"
    if _STRICT_VALIDATION_ONLY:
        env.pop("GAGC_TEST_DIR", None)
        env.pop("GR_TEST_DATA", None)
    if _DATA_DIR:
        env["GAGC_DATA_DIR"] = _DATA_DIR
    if _TEST_DIR and not _STRICT_VALIDATION_ONLY:
        env["GAGC_TEST_DIR"] = _TEST_DIR
    if cfg.gpu_ids or cfg.num_gpus > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id_for_slot(slot_id))
    env["GAGC_SLOT_ID"] = str(slot_id)
    if _BENCHMARK_MODE == "kuairec":
        if _GR_TRAIN_DATA:
            env["GR_TRAIN_DATA"] = _GR_TRAIN_DATA
        eval_data = _GR_VAL_DATA or ("" if _STRICT_VALIDATION_ONLY else _GR_TEST_DATA)
        if eval_data:
            env["GR_TEST_DATA"] = eval_data
        env.setdefault("GR_NUM_EPOCHS", "2")
    elif _BENCHMARK_MODE == "amazon_reviews":
        env.setdefault("SASREC_EPOCHS", "200")
        env.setdefault("GRU4REC_EPOCHS", "200")
        env.setdefault("BERT4REC_EPOCHS", "200")
        env.setdefault("NEXTITNET_EPOCHS", "200")
        env.setdefault("HSTU_EPOCHS", "200")
    return env, cpu_set


def _uses_claude_code_backend(spec: MutationSpec) -> bool:
    return str(spec.extra.get("implementation_backend", "")).strip().lower() == "claude_code"


def _has_inline_mutation_payload(spec: MutationSpec) -> bool:
    return bool(
        spec.code_edits
        or spec.code_content.strip()
        or spec.code_diff.strip()
        or spec.extra.get("predict_code_edits")
        or str(spec.extra.get("predict_code_content") or "").strip()
    )


def _claude_code_backend_fields_for_arm(arm: str) -> dict[str, Any]:
    if _BENCHMARK_MODE == "e2_shared":
        return {
            "implementation_backend": "endpoint2_native_e2",
            "implementation_prompt": (
                "Implement the selected hypothesis in the shared E2 four-file program. "
                "best.py is a JSON source bundle, not an executable training script. "
                "Preserve unchanged parent mechanisms and the frozen E2 evaluator."
            ),
            "allowed_files": ["models.py", "diagnosis_tools.py", "main_code.py:train_update_kernel"],
        }
    if _BENCHMARK_MODE == "e1_ml1m":
        return {
            "implementation_backend": "endpoint2_native_recbole",
            "implementation_prompt": (
                "Implement the selected causal hypothesis in FreshCandidateModel. "
                "best.py is the RecBole class source, not a trainer. The shared RO "
                "native runner owns data, optimizer, stopping, and full-sort NDCG@10. "
                "Preserve the active parent outside the declared intervention."
            ),
            "allowed_files": ["best.py"],
        }
    if _BENCHMARK_MODE == "kuairec":
        if arm != "change_decoder_backbone":
            return {}
        return {
            "implementation_backend": "claude_code",
            "implementation_prompt": (
                "Implement one GR decoder-backbone architecture jump. Prefer replacing the Transformer decoder "
                "with an LSTM decoder while preserving the Seq2Seq interface, metric printing contract, data "
                "loading, dynamic vocabulary, loss computation, and 2-epoch proxy behavior. Keep Transformer-specific "
                "code available as fallback when practical. This jump is allowed to underperform before retuning; "
                "focus on a correct runnable new basin."
            ),
            "allowed_files": [
                "train.py",
                "model/decoder.py",
                "model/transformer.py",
                "model/encoder.py",
            ],
            "claude_attempts": 3,
        }

    if _BENCHMARK_MODE != "amazon_reviews":
        return {}
    return {
        "implementation_backend": "claude_code",
        "implementation_prompt": (
            "Implement one isolated Amazon Reviews sequence-recommendation trial for the selected RecHarness low-level edit arm. "
            "Treat the arm/dimension as the specific edit scope: make one small runnable change aligned with code_hint, "
            "hypothesis, delta, ExperimentSkill lessons, and existing train.py/predict.py code. Composite arms such as "
            "tune_lr_batch_scheduler and tune_dropout_wd may adjust their coupled low-level knobs together, but should "
            "avoid broad unrelated rewrites. Preserve the training entrypoint, per-dataset checkpoint layout when present, "
            "predict.py predict(user_id, history, candidates, dataset_name=None) contract, validation-only search policy, "
            "and benchmark metric printing/parsing behavior. Do not use held-out test metrics during search."
        ),
        "allowed_files": ["train.py", "predict.py"],
        "claude_attempts": 3,
    }


def _validate_trial_syntax_and_contract(train_path: str, predict_path: str) -> str | None:
    for path, label in ((train_path, "train.py"), (predict_path, "predict.py")):
        if not os.path.exists(path):
            continue
        err = _syntax_check_python(path, label)
        if err:
            return err
    return _contract_check_trial_files(train_path, predict_path)


def _restore_provisional_incumbent(snapshot: dict[str, Any]) -> str | None:
    """Restore best.py/predict.py/train.py from a provisional-jump snapshot."""
    script_path = str(snapshot.get("script_path") or "")
    predict_path = str(snapshot.get("predict_path") or "")
    train_path = str(snapshot.get("train_path") or "")
    best_code = snapshot.get("best_code")
    predict_code = snapshot.get("predict_code")
    train_code = snapshot.get("train_code")
    if not script_path or not isinstance(best_code, str):
        return "missing script_path or best_code in provisional snapshot"
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(best_code)
        if train_path and isinstance(train_code, str) and train_code:
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(train_code)
        if predict_path and isinstance(predict_code, str):
            with open(predict_path, "w", encoding="utf-8") as f:
                f.write(predict_code)
    except OSError as exc:
        return str(exc)
    return None


def _normalise_trial_results_input(
    trial_results_json: str,
    expected_group_id: str = "",
    allow_durable_fallback: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Prefer the last raw execute_trial_group payload when LLM passed a lossy summary.

    When expected_group_id is provided, durable fallback results whose trial_group_id
    does not match are rejected to prevent cross-group contamination. Search-only
    callers disable durable fallback because `/logs` is agent-writable; a trusted
    resume manifest is required before that mode can recover across processes.
    """
    try:
        parsed = json.loads(trial_results_json) if trial_results_json.strip() else []
    except Exception:
        parsed = []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        parsed = []

    def _has_full_spec(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        spec = result.get("spec")
        return isinstance(spec, dict) and ("arm" in spec or "code_edits" in spec or "code_content" in spec)

    parsed_has_full_specs = bool(parsed) and all(_has_full_spec(r) for r in parsed)
    if parsed_has_full_specs:
        if expected_group_id:
            parsed_group_ids = {str(r.get("trial_group_id", "")) for r in parsed if isinstance(r, dict)}
            if parsed_group_ids != {expected_group_id}:
                return [], False
        return parsed, False

    if allow_durable_fallback:
        durable_results = _read_latest_trial_results_log()
        if durable_results:
            if expected_group_id:
                durable_group_id = durable_results[0].get("trial_group_id", "") if durable_results else ""
                if durable_group_id and durable_group_id != expected_group_id:
                    # Group ID mismatch — reject stale fallback to avoid cross-group contamination
                    durable_results = []
            if durable_results:
                return durable_results, True

    if _LAST_TRIAL_RESULTS:
        if expected_group_id:
            mem_group_id = _LAST_TRIAL_RESULTS[0].get("trial_group_id", "") if _LAST_TRIAL_RESULTS else ""
            if mem_group_id and mem_group_id != expected_group_id:
                return [], False
        return deepcopy(_LAST_TRIAL_RESULTS), True
    if expected_group_id:
        return [], False
    return parsed, False


def _trial_group_wall_time_secs(results: list[dict[str, Any]]) -> tuple[float, str]:
    """Return the budget cost for one executed trial group as the sum of the
    parallel members' individual runtimes.

    `global_budget` is denominated in compute-seconds, so a parallel group is
    charged the total compute it consumed -- the sum of each trial's
    `wall_time_secs` across all members -- not the single batch wall-clock time.
    Older logs that recorded only a group wall time (no per-member runtimes)
    fall back to that batch wall time as the best available conservative lower
    bound, since the per-member split cannot be reconstructed.
    """
    member_times: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            member_time = float(result.get("wall_time_secs", 0.0) or 0.0)
        except (TypeError, ValueError):
            member_time = 0.0
        if member_time > 0.0:
            member_times.append(member_time)
    if member_times:
        return sum(member_times), "sum_member_wall_time_secs"
    # Fallback: no per-member runtimes recorded; use the batch wall time if present.
    group_times: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            group_time = float(result.get("group_wall_time_secs", 0.0) or 0.0)
        except (TypeError, ValueError):
            group_time = 0.0
        if group_time > 0.0:
            group_times.append(group_time)
    if group_times:
        return max(group_times), "group_wall_time_secs_fallback"
    return 0.0, "missing_wall_time"


def _contains_heldout_feedback(result: dict[str, Any]) -> bool:
    """Return True when a search result carries non-empty held-out feedback."""
    for key in ("test_score", "test_metrics"):
        value = result.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _proposal_member_identity(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Stable fields that identify membership in a frozen proposal group."""
    return (
        str(spec.get("arm", spec.get("dimension", ""))),
        str(spec.get("dimension", "")),
        bool(spec.get("is_jumping")),
        bool(spec.get("is_baseline")),
        str(spec.get("implementation_backend", "")),
        tuple(str(path) for path in (spec.get("allowed_files") or [])),
    )


def _strict_group_results_error(results: list[dict[str, Any]]) -> str | None:
    """Validate that state-changing search inputs are the last frozen group."""
    if not results:
        return "search-only operation has no cached trial results"
    if any(_contains_heldout_feedback(result) for result in results):
        return "search-only operation received held-out feedback"
    if not _LAST_PROPOSED_CANDIDATES:
        return "search-only operation has no frozen proposal group"

    expected_members = [
        _proposal_member_identity(spec) for spec in _LAST_PROPOSED_CANDIDATES
    ]
    actual_members = [
        _proposal_member_identity(result.get("spec") or {}) for result in results
    ]
    if actual_members != expected_members:
        return "search-only results do not match the frozen proposal group"

    group_ids = {str(result.get("trial_group_id") or "") for result in results}
    if len(group_ids) != 1 or "" in group_ids:
        return "search-only results must share one non-empty trial_group_id"
    result_group_id = next(iter(group_ids))
    if not _LAST_TRIAL_GROUP_ID or result_group_id != _LAST_TRIAL_GROUP_ID:
        return "search-only results are not the latest executed trial group"

    jumping_dims = _active_jumping_dims()
    is_jumping_group = any(
        bool((result.get("spec") or {}).get("is_jumping"))
        or str((result.get("spec") or {}).get("arm", "")) in jumping_dims
        or str((result.get("spec") or {}).get("dimension", "")) in jumping_dims
        for result in results
    )
    is_baseline_group = all(
        _is_baseline_spec_data(result.get("spec") or {}) for result in results
    )
    required_group_size = _SERVER_CONFIG.logical_group_size
    if (
        required_group_size is not None
        and not is_jumping_group
        and not is_baseline_group
        and len(results) != required_group_size
    ):
        return "normal search-only results do not preserve logical_group_size"
    return None


def _latest_trial_results_path() -> str:
    return os.path.join(_LOGS_ROOT, "trials", "latest_results.json") if _LOGS_ROOT else ""


def _write_latest_trial_results_log(payload: dict[str, Any]) -> None:
    if not _LOGS_ROOT:
        return
    _write_json_log("trials/latest_results.json", payload)



def _tool_error_results(error: str, trial_group_id: str, script_path: str) -> str:
    global _LAST_TRIAL_RESULTS, _LAST_TRIAL_GROUP_ID
    error_result = _tool_error_result(error, trial_group_id=trial_group_id)
    _LAST_TRIAL_RESULTS = [deepcopy(error_result)]
    _LAST_TRIAL_GROUP_ID = trial_group_id
    _write_latest_trial_results_log({
        "trial_group_id": trial_group_id,
        "timestamp": _utc_now_iso(),
        "script_path": script_path,
        "results": [error_result],
    })
    return json.dumps([error_result], indent=2)


def _parse_agent_hypotheses(hypotheses_json: str) -> tuple[dict[str, str], str | None]:
    raw = (hypotheses_json or "").strip()
    if not raw:
        return {}, None
    if len(raw) > _MAX_AGENT_HYPOTHESES_JSON_CHARS:
        return {}, f"hypotheses_json too large; keep it under {_MAX_AGENT_HYPOTHESES_JSON_CHARS} chars"
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {}, f"invalid hypotheses_json: {exc}"
    if isinstance(payload, list):
        payload = {
            str(item.get("arm") or item.get("dimension") or idx): item.get("hypothesis") or item.get("agent_hypothesis") or item.get("note") or ""
            for idx, item in enumerate(payload)
            if isinstance(item, dict)
        }
    if not isinstance(payload, dict):
        return {}, "invalid hypotheses_json: expected object mapping arm/dimension to short hypothesis"
    parsed: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            text = value.get("hypothesis") or value.get("agent_hypothesis") or value.get("note") or ""
        else:
            text = value
        text = " ".join(str(text).split())
        if not text:
            continue
        if len(text) > _MAX_AGENT_HYPOTHESIS_CHARS:
            return {}, f"hypothesis for {key} too long; keep each under {_MAX_AGENT_HYPOTHESIS_CHARS} chars"
        parsed[str(key)] = text
    return parsed, None


def _memory_context_for_claude() -> dict[str, Any]:
    state = None
    raw = ""
    if _STORE is not None:
        try:
            item = _STORE.get(_STORE_NAMESPACE, _STORE_KEY)
            raw = item.value["content"] if item else ""
        except Exception:
            raw = ""
    if raw:
        try:
            state = ThompsonState.from_json(raw)
        except Exception:
            state = None
    if state is None:
        return {}
    return {
        "experiment_skill": state.experiment_skill,
        "recent_text_gradients": list(state.recent_text_gradients[-_MAX_TEXT_GRADIENTS:]),
        "failure_memory": list(state.failure_memory[-3:]),
        "skill_notes": state.skill_notes,
    }


def _prepare_specs_for_execution(
    specs_json: str,
    hypotheses_json: str,
    trial_group_id: str,
    script_path: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    raw = (specs_json or "").strip()
    use_last = raw in {"", "__LAST_PROPOSED__", "LAST_PROPOSED"}
    if use_last:
        if not _LAST_PROPOSED_CANDIDATES:
            return None, "no cached candidates; call propose_action_group before execute_trial_group('__LAST_PROPOSED__')"
        specs = deepcopy(_LAST_PROPOSED_CANDIDATES)
    else:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            return None, f"invalid specs_json: {exc}"
        if isinstance(parsed, dict) and parsed.get("use_last_proposed"):
            if not _LAST_PROPOSED_CANDIDATES:
                return None, "no cached candidates; call propose_action_group before use_last_proposed"
            specs = deepcopy(_LAST_PROPOSED_CANDIDATES)
        else:
            specs = parsed

    if not isinstance(specs, list) or not specs or not all(isinstance(spec, dict) for spec in specs):
        return None, "invalid specs_json: expected a non-empty JSON array of objects or __LAST_PROPOSED__"

    hypotheses, err = _parse_agent_hypotheses(hypotheses_json)
    if err:
        return None, err
    memory_context = _memory_context_for_claude()
    prepared: list[dict[str, Any]] = []
    for spec in specs:
        item = deepcopy(spec)
        is_claude_spec = str(item.get("implementation_backend", "")).strip().lower() == "claude_code"
        if use_last or is_claude_spec:
            item["code_edits"] = []
            item["code_diff"] = ""
            item["code_content"] = ""
            item["predict_code_edits"] = []
            item["predict_code_content"] = ""
        arm = str(item.get("arm") or item.get("dimension") or "")
        primary_dim = str(item.get("dimension") or "")
        note = hypotheses.get(arm) or hypotheses.get(primary_dim)
        if note:
            base = str(item.get("hypothesis") or item.get("code_hint") or "")
            item["agent_hypothesis"] = note
            item["hypothesis"] = (base + "; Agent hypothesis: " + note) if base else note
        prompt = " ".join(str(item.get("implementation_prompt") or "").split())
        if len(prompt) > _MAX_CLAUDE_IMPLEMENTATION_PROMPT_CHARS:
            item["implementation_prompt"] = prompt[:_MAX_CLAUDE_IMPLEMENTATION_PROMPT_CHARS].rstrip() + " ..."
            item["implementation_prompt_truncated"] = True
        if memory_context:
            item["gagc_memory_context"] = memory_context
        prepared.append(item)
    _write_json_log(f"trials/{trial_group_id}_prepared_specs.json", {
        "trial_group_id": trial_group_id,
        "timestamp": _utc_now_iso(),
        "script_path": script_path,
        "used_last_proposed": use_last,
        "hypothesis_keys": sorted(hypotheses),
        "specs": prepared,
    })
    return prepared, None


def _read_latest_trial_results_log() -> list[dict[str, Any]]:
    path = _latest_trial_results_path()
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results") if isinstance(payload, dict) else None
        return deepcopy(results) if isinstance(results, list) else []
    except Exception:
        return []


def _select_winner_index(results: list[dict[str, Any]]) -> int | None:
    winner = _select_winner_by_benchmark(results)
    if winner is None:
        return None
    for idx, result in enumerate(results):
        if result is winner or result == winner:
            return idx
    return None


def _hydrate_trial_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge stripped tool-output results with durable/in-memory raw payloads.

    execute_trial_group intentionally strips large mutated_* fields from the JSON
    returned to the LLM. promote_winner needs those fields, so when a caller passes
    stripped results we hydrate them from logs/trials/latest_results.json or the
    in-memory cache using trial_group_id + trial_id/index.
    """
    if not results:
        return results
    if all(isinstance(r, dict) and r.get("mutated_code_content") for r in results):
        return results

    group_id = ""
    for result in results:
        if isinstance(result, dict) and result.get("trial_group_id"):
            group_id = str(result.get("trial_group_id"))
            break
    if not group_id:
        return results

    raw_candidates = _read_latest_trial_results_log()
    if not raw_candidates or str(raw_candidates[0].get("trial_group_id", "")) != group_id:
        raw_candidates = deepcopy(_LAST_TRIAL_RESULTS)
    if not raw_candidates or str(raw_candidates[0].get("trial_group_id", "")) != group_id:
        return results

    raw_by_trial_id = {
        int(raw.get("trial_id", idx)): raw
        for idx, raw in enumerate(raw_candidates)
        if isinstance(raw, dict)
    }
    hydrated: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            hydrated.append(result)
            continue
        trial_id = int(result.get("trial_id", idx))
        raw = raw_by_trial_id.get(trial_id)
        if raw is None:
            hydrated.append(result)
            continue
        merged = deepcopy(raw)
        merged.update(result)
        for large_key in ("mutated_code_content", "mutated_predict_content"):
            if large_key in raw and large_key not in result:
                merged[large_key] = raw[large_key]
        hydrated.append(merged)
    return hydrated


def _apply_spec_to_contents(
    base_code: str,
    base_predict: str,
    spec_data: dict[str, Any],
) -> tuple[str, str, str | None]:
    """Recreate the winner's train/predict contents from incumbent files and spec."""
    # Promotion can be called with externally-shaped spec dicts that may omit
    # bookkeeping fields like estimated_cost_secs. Those fields are unused by
    # the edit application below, so fill safe defaults rather than crash.
    # 3h matches the configured long-running trial ceiling — never under-budget.
    safe_spec = dict(spec_data)
    safe_spec.setdefault("dimension", "")
    safe_spec.setdefault("delta", 0.0)
    safe_spec.setdefault("estimated_cost_secs", 3 * 3600.0)
    safe_spec.setdefault("code_diff", "")
    spec = MutationSpec(**safe_spec)
    train_code = base_code
    predict_code = base_predict

    if spec.code_edits:
        train_code, err = _apply_edits(train_code, spec.code_edits)
        if err:
            return base_code, base_predict, f"train.py promotion edit failed: {err}"
    elif spec.code_content.strip():
        train_code = spec.code_content
    elif spec.code_diff.strip():
        return base_code, base_predict, "code_diff promotion is not supported; use code_edits or code_content"

    predict_edits = spec.extra.get("predict_code_edits") or []
    predict_content = str(spec.extra.get("predict_code_content") or "")
    if predict_edits:
        predict_code, err = _apply_edits(predict_code, predict_edits)
        if err:
            return base_code, base_predict, f"predict.py promotion edit failed: {err}"
    elif predict_content.strip():
        predict_code = predict_content

    return train_code, predict_code, None


def _error_counts(results: list[dict]) -> dict[str, int]:
    counts = {
        "timeout": 0,
        "oom": 0,
        "crash": 0,
        "edit_failed": 0,
        "implementation_failure": 0,
    }
    for result in results:
        failure = _classify_failure(result) if not _is_valid_trial(result) else "none"
        if failure in counts:
            counts[failure] += 1
        elif failure == "provider_failure":
            counts[failure] = counts.get(failure, 0) + 1
    return counts


def _tool_error_result(error: str, *, trial_group_id: str = "") -> dict[str, Any]:
    """Return a TrialResult-shaped tool error without raising through LangGraph."""
    return {
        "spec": {
            "dimension": "__tool_error__",
            "arm": "__tool_error__",
            "delta": 0.0,
            "estimated_cost_secs": 0.0,
            "code_diff": "",
            "code_edits": [],
            "code_content": "",
            "is_tool_error": True,
        },
        "trial_group_id": trial_group_id,
        "trial_id": 0,
        "wall_time_secs": 0.0,
        "timed_out": False,
        "oom": False,
        "val_score": _CRASH_SCORE,
        "val_metrics": None,
        "test_score": None,
        "test_metrics": None,
        "convergence_trace": [],
        "error_message": error,
        "stdout_tail": "",
        "stderr_tail": "",
        "tool_error": True,
    }
_DATA_DIR: str = "./input/trainval"
_TEST_DIR: str = "./input/trainval"

# ---------------------------------------------------------------------------
# Dimension / arm definitions
# ---------------------------------------------------------------------------

_DIMENSION_HINTS: dict[str, str] = {
    "tune_lr": "Change LR in train.py. delta > 0 means increase, < 0 means decrease.",
    "tune_batch_size": "Change BATCH_SIZE in train.py. delta > 0 means larger batch, < 0 means smaller.",
    "tune_dropout": "Change DROPOUT in train.py. delta > 0 means more regularisation, < 0 means less.",
    "tune_weight_decay": "Change weight_decay in train.py. delta > 0 means stronger L2, < 0 means weaker.",
    "tune_embedding_dim": "Change the embedding dimension / hidden size in train.py. delta > 0 means wider, < 0 means narrower.",
    "tune_num_layers": "Change the number of transformer/recurrent layers in train.py. delta > 0 means deeper, < 0 means shallower.",
    "tune_num_heads": "Change the number of attention heads in train.py. delta > 0 means more heads, < 0 means fewer.",
    "tune_seq_len": "Change maximum input sequence length in train.py. delta > 0 means longer context, < 0 means shorter.",
    "change_loss_function": "Switch the loss function in train.py, for example BCE, BPR, sampled-softmax, or hybrid ranking objectives. This is a high-risk jump arm.",
    "change_architecture": "Change model architecture within the template family while preserving benchmark contracts. This is a high-risk jump arm.",
    "change_pooling": "Change sequence pooling / user representation strategy, for example last-token, mean, attention, or recency-weighted pooling.",
    "add_features": "Add lightweight side-information or priors, such as item popularity, position, time gap, or dataset-specific score calibration.",
    "add_lr_scheduler": "Add or change the learning-rate scheduler, for example warmup + cosine, step decay, or cyclic schedule.",
    "tune_lr_batch": "Jointly tune learning rate and batch size because they often interact through optimization stability.",
    "tune_dropout_wd": "Jointly tune dropout and weight_decay as coupled regularization knobs.",
    "tune_lr_batch_scheduler": "Jointly tune LR, batch size, and LR scheduler as a coupled optimization setup.",
}

_ALL_DIMENSIONS: list[str] = [
    "tune_lr", "tune_batch_size", "tune_dropout", "tune_weight_decay",
    "tune_embedding_dim", "tune_num_layers", "tune_num_heads", "tune_seq_len",
    "change_loss_function", "change_architecture", "change_pooling",
    "add_features", "add_lr_scheduler",
]

_ALL_ARMS: list[str] = [
    "tune_lr_batch_scheduler",
    "tune_dropout_wd",
    "tune_embedding_dim", "tune_num_layers", "tune_num_heads", "tune_seq_len",
    "change_loss_function", "change_architecture", "change_pooling",
    "add_features",
    "tune_lr", "tune_batch_size", "tune_dropout", "tune_weight_decay", "add_lr_scheduler",
]

_FIXED_DISCOVERY_ARM = None

_AMAZON_JUMP_ELIGIBLE_STRATEGY_ARMS: set[str] = set()

_EXPLOITING_ARMS: list[str] = [a for a in _ALL_ARMS if a not in JUMPING_DIMS]

# Task vocabulary only. Routing, group advantages, posterior updates, jump
# eligibility, promotion, and SkillOpt remain the upstream RecHarness algorithms.
_E1_HINTS = {
    "encoder_representation": "Change how the encoder represents a user's implicit interaction vector.",
    "decoder_scoring": "Change the causal mapping from latent user state to item scores.",
    "input_denoising": "Improve training-view corruption or reconstruction without using dev interactions.",
    "latent_regularization": "Change the latent prior or posterior regularization with a clear effect hypothesis.",
    "history_aggregation": "Change aggregation or weighting of train-only user history; this task is not sequential.",
    "popularity_correction": "Correct item-popularity bias in scoring using train-only evidence.",
    "change_loss_function": "Propose a coherent reconstruction/ranking objective change within the fixed trainer ABI.",
    "change_architecture": "Propose a causal architecture rewrite; preserve the fixed RecBole class and evaluator ABI.",
}

# ---------------------------------------------------------------------------

_GR_DIMENSION_HINTS: dict[str, str] = {
    "tune_q_start":          "Change Q_START (dynamic vocab percentile start). delta > 0 means finer-grained vocab top-end, < 0 means coarser.",
    "tune_q_end":            "Change Q_END (dynamic vocab percentile floor). delta > 0 raises floor (smaller vocab), < 0 lowers it (larger vocab).",
    "tune_q_decay":          "Change Q_DECAY (vocab quantile decay rate). delta > 0 means slower decay (larger vocab), < 0 means faster decay (smaller vocab).",
    "tune_window_size":      "Change WINDOW_SIZE for soft-argmax decoding. delta > 0 means wider window (smoother), < 0 means narrower (sharper).",
    "tune_cls_weight":       "Change CLS_WEIGHT (cross-entropy loss weight). delta > 0 means stronger ranking signal, < 0 means weaker.",
    "tune_huber_weight":     "Change HUBER_WEIGHT (regression loss weight). delta > 0 means stronger regression, < 0 means weaker.",
    "toggle_embedding_mixup":"Toggle USE_MIXUP flag as a regularization/training trick. delta > 0 enables embedding mixup (soft token embeddings), < 0 disables it.",
    "change_curriculum_type":"Change curriculum learning schedule type (linear / exp / sigmoid) as a teacher-forcing schedule tweak. delta > 0 moves to more aggressive decay, < 0 to gentler.",
    "change_decoder_backbone":"Change the GR decoder backbone (for example Transformer decoder → LSTM/GRU/TCN). This is a high-risk architecture jumping arm that must retune for several rounds before judging.",
    "tune_lr":               "Change LR in train.py. delta > 0 means increase, < 0 means decrease.",
    "tune_batch_size":       "Change BATCH_SIZE in train.py. delta > 0 means larger batch, < 0 means smaller.",
    "tune_dropout":          "Change DROPOUT in train.py. delta > 0 means more regularisation, < 0 means less.",
    "tune_dec_layers":       "Change DEC_LAYERS (number of transformer decoder layers). delta > 0 means deeper, < 0 means shallower.",
    "tune_hidden_dim":       "Change HIDDEN_DIM / FEAT_DIM in train.py. delta > 0 means wider model, < 0 means narrower.",
    "tune_num_heads":        "Change N_HEAD in train.py. delta > 0 means more heads, < 0 means fewer.",
    "add_lr_scheduler":      "Add or change the LR scheduler in train.py (warmup + cosine / step decay / cyclic).",
    "tune_optimizer_schedule":"Jointly tune LR, batch size, and scheduler because optimizer stability is coupled.",
    "tune_loss_balance":     "Jointly tune CLS_WEIGHT and HUBER_WEIGHT because they define the CE/regression loss balance.",
    "tune_vocab_quantization":"Jointly tune Q_START, Q_END, and Q_DECAY because they define the dynamic vocabulary granularity.",
    "tune_transformer_capacity":"Jointly tune HIDDEN_DIM/FEAT_DIM, N_HEAD, and DEC_LAYERS while preserving divisibility constraints.",
}

_GR_ALL_DIMENSIONS: list[str] = list(_GR_DIMENSION_HINTS.keys())

# GR arms = composite arms plus standalone dims that are not subsumed.
_GR_SUBSUMED_DIMS: set[str] = {
    dim for dims in GR_COMPOSITE_ARMS.values() for dim in dims
}
_GR_ALL_ARMS: list[str] = list(GR_COMPOSITE_ARMS.keys()) + [
    dim for dim in _GR_ALL_DIMENSIONS
    if dim not in _GR_SUBSUMED_DIMS and dim not in GR_COMPOSITE_ARMS
]
_GR_EXPLOITING_ARMS: list[str] = [a for a in _GR_ALL_ARMS if a not in GR_JUMPING_DIMS]

# ---------------------------------------------------------------------------

_DIAG_RULES: list[tuple[str, str, str]] = [
    ("nan",              "tune_lr",             "NaN detected — LR likely too high, decrease LR"),
    ("nan",              "change_loss_function", "NaN detected — BCE numerically unstable, switch to BPR or sampled softmax"),
    ("inf",              "tune_lr",             "Inf gradient — decrease LR to stabilise training"),
    ("out of memory",    "tune_batch_size",     "OOM — reduce batch size"),
    ("out of memory",    "tune_embedding_dim",  "OOM — reduce embedding dimension"),
    ("out of memory",    "tune_num_layers",     "OOM — reduce number of layers"),
    ("timeout",          "tune_num_layers",     "Timeout — model too deep, reduce layers"),
    ("timeout",          "tune_seq_len",        "Timeout — sequence too long, reduce seq_len"),
    ("overfit",          "tune_dropout",        "Overfitting — increase dropout"),
    ("overfit",          "tune_weight_decay",   "Overfitting — increase weight decay"),
    ("overfit",          "tune_num_layers",     "Overfitting — reduce model capacity, decrease layers"),
    ("val.*lower.*train", "tune_dropout",       "Train/val gap — increase dropout to reduce overfitting"),
    ("not.*decreas",     "add_lr_scheduler",    "Loss plateau — add warmup + cosine scheduler"),
    ("plateau",          "add_lr_scheduler",    "Loss plateau — add warmup + cosine scheduler"),
    ("slow.*converg",    "tune_lr",             "Slow convergence — try increasing LR"),
    ("slow.*converg",    "add_lr_scheduler",    "Slow convergence — add lr scheduler with warmup"),
    ("oscillat",         "tune_lr",             "Loss oscillating — decrease LR for stability"),
    ("oscillat",         "change_loss_function","Loss oscillating — switch to smoother loss function"),
    ("underfit",         "tune_num_layers",     "Underfitting — increase model depth"),
    ("underfit",         "tune_embedding_dim",  "Underfitting — increase embedding dimension"),
]

_GR_DIAG_RULES: list[tuple[str, str, str]] = [
    ("nan",              "tune_lr",           "NaN detected — LR too high, decrease LR"),
    ("nan",              "tune_cls_weight",   "NaN in CE loss — reduce cls_weight"),
    ("inf",              "tune_lr",           "Inf gradient — decrease LR to stabilise training"),
    ("out of memory",    "tune_batch_size",   "OOM — reduce BATCH_SIZE"),
    ("out of memory",    "tune_hidden_dim",   "OOM — reduce HIDDEN_DIM/FEAT_DIM"),
    ("out of memory",    "tune_dec_layers",   "OOM — reduce DEC_LAYERS"),
    ("timeout",          "tune_dec_layers",   "Timeout — model too deep, reduce DEC_LAYERS"),
    ("timeout",          "tune_batch_size",   "Timeout — increase BATCH_SIZE for faster epoch"),
    ("overfit",          "tune_dropout",      "Overfitting — increase DROPOUT"),
    ("overfit",          "tune_huber_weight", "Overfitting on regression — reduce HUBER_WEIGHT"),
    ("not.*decreas",     "add_lr_scheduler",  "Loss plateau — add warmup + cosine LR scheduler"),
    ("plateau",          "add_lr_scheduler",  "Loss plateau — add LR scheduler"),
    ("slow.*converg",    "tune_lr",           "Slow convergence — try increasing LR"),
    ("slow.*converg",    "add_lr_scheduler",  "Slow convergence — add scheduler with warmup"),
    ("oscillat",         "tune_lr",           "Loss oscillating — decrease LR for stability"),
    ("oscillat",         "tune_cls_weight",   "CE loss oscillating — lower cls_weight"),
    ("mae.*high",        "tune_window_size",  "MAE high — try wider window_size for softer decoding"),
    ("mae.*high",        "tune_huber_weight", "MAE high — increase HUBER_WEIGHT for stronger regression"),
    ("xauc.*low",        "tune_cls_weight",   "xAUC low — increase cls_weight for better ranking"),
    ("xauc.*low",        "tune_q_start",      "xAUC low — try adjusting vocab resolution via q_start"),
    ("underfit",         "tune_dec_layers",   "Underfitting — increase DEC_LAYERS"),
    ("underfit",         "tune_hidden_dim",   "Underfitting — increase HIDDEN_DIM"),
]


def _active_arms() -> list[str]:
    if _BENCHMARK_MODE == "e2_shared":
        from gagc.e2_task import HINTS
        return list(HINTS)
    if _BENCHMARK_MODE == "e1_ml1m":
        return list(_E1_HINTS)
    return _GR_ALL_ARMS if _BENCHMARK_MODE == "kuairec" else _ALL_ARMS


def _active_exploiting_arms() -> list[str]:
    if _BENCHMARK_MODE == "e2_shared":
        return [arm for arm in _active_arms() if arm not in JUMPING_DIMS]
    if _BENCHMARK_MODE == "e1_ml1m":
        return [arm for arm in _E1_HINTS if arm not in JUMPING_DIMS]
    return _GR_EXPLOITING_ARMS if _BENCHMARK_MODE == "kuairec" else _EXPLOITING_ARMS


def _active_fixed_discovery_arm() -> str | None:
    if _BENCHMARK_MODE == "amazon_reviews":
        return _FIXED_DISCOVERY_ARM
    return None


def _active_dimensions() -> list[str]:
    """Original flat dimension list (for backward compat where needed)."""
    if _BENCHMARK_MODE == "e2_shared":
        return [arm for arm in _active_arms() if arm not in _active_composite_arms()]
    if _BENCHMARK_MODE == "e1_ml1m":
        return list(_E1_HINTS)
    return _GR_ALL_DIMENSIONS if _BENCHMARK_MODE == "kuairec" else _ALL_DIMENSIONS


def _active_dimension_hints() -> dict[str, str]:
    if _BENCHMARK_MODE == "e2_shared":
        from gagc.e2_task import HINTS
        return HINTS
    if _BENCHMARK_MODE == "e1_ml1m":
        return _E1_HINTS
    return _GR_DIMENSION_HINTS if _BENCHMARK_MODE == "kuairec" else _DIMENSION_HINTS


def _active_diag_rules() -> list[tuple[str, str, str]]:
    if _BENCHMARK_MODE == "e2_shared":
        return [rule for rule in _DIAG_RULES if rule[1] in _active_dimensions()]
    if _BENCHMARK_MODE == "e1_ml1m":
        return []  # Sequence-specific keyword-to-arm rules do not apply to E1.
    return _GR_DIAG_RULES if _BENCHMARK_MODE == "kuairec" else _DIAG_RULES


def _active_jumping_dims() -> set[str]:
    return GR_JUMPING_DIMS if _BENCHMARK_MODE == "kuairec" else JUMPING_DIMS


def _active_jump_eligible_strategy_arms() -> set[str]:
    if _BENCHMARK_MODE == "amazon_reviews":
        return set(_AMAZON_JUMP_ELIGIBLE_STRATEGY_ARMS)
    return set(_active_jumping_dims())


def _active_mutex_groups() -> list[set[str]]:
    return GR_MUTEX_GROUPS if _BENCHMARK_MODE == "kuairec" else MUTEX_GROUPS


def _active_composite_arms() -> dict[str, list[str]]:
    if _BENCHMARK_MODE == "e2_shared":
        from gagc.e2_task import COMPOSITES
        return COMPOSITES
    if _BENCHMARK_MODE == "e1_ml1m":
        return {}
    return GR_COMPOSITE_ARMS if _BENCHMARK_MODE == "kuairec" else COMPOSITE_ARMS


def _expand_arm(arm: str) -> list[str]:
    """Expand a composite arm to its component dimensions."""
    composites = _active_composite_arms()
    return composites.get(arm, [arm])


def _routing_policy_enabled() -> str:
    """Return the active routing policy, allowing env override for ablations."""
    policy = os.getenv("GAGC_ROUTING_POLICY", _ROUTING_POLICY).strip().lower()
    aliases = {
        "ts": "thompson",
        "on": "thompson",
        "true": "thompson",
        "1": "thompson",
        "off": "textual_gradient",
        "false": "textual_gradient",
        "0": "textual_gradient",
        "memory": "textual_gradient",
        "llm": "textual_gradient",
        "llm_memory": "textual_gradient",
        "textual": "textual_gradient",
        "textual-gradient": "textual_gradient",
    }
    return aliases.get(policy, policy)


def _is_textual_gradient_policy() -> bool:
    return _routing_policy_enabled() == "textual_gradient"


def _configured_trial_floor_secs() -> float:
    if _BENCHMARK_MODE == "e1_ml1m":
        return 1200.0
    default = 6400.0 if _BENCHMARK_MODE == "kuairec" else 10800.0
    raw = os.getenv("GAGC_TRIAL_SECS", "").strip()
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


def default_experiment_skill_for_policy() -> str:
    if not _is_textual_gradient_policy():
        from gagc.state import _DEFAULT_EXPERIMENT_SKILL
        return _DEFAULT_EXPERIMENT_SKILL
    return """\
# RecHarness Experiment Skill

## Objective
Use benchmark val_score as the search and winner criterion.
Do not use held-out test metrics during search.
Textual-gradient routing chooses arms from experiment memory; this skill guides both arm selection and code edits without using α/β ranking.

## Recent Patch Lessons
No learned patch lessons yet.

## Concrete Avoids
No concrete patch-level avoids yet.
"""


def _parse_requested_arms(text: str) -> list[str]:
    """Parse LLM-requested arm names for the textual-gradient ablation."""
    if not text or not text.strip():
        return []
    valid = set(_active_arms())
    try:
        data = json.loads(text)
    except Exception:
        data = text

    raw_items: list[Any] = []
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ("arms", "selected_arms", "dimensions", "selected_dimensions"):
            value = data.get(key)
            if isinstance(value, list):
                raw_items = value
                break
        if not raw_items:
            raw_items = [data]
    elif isinstance(data, str):
        raw_items = re.split(r"[,\n\s]+", data)

    parsed: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            candidate = str(item.get("arm") or item.get("dimension") or item.get("name") or "").strip()
        else:
            candidate = str(item).strip()
        if candidate in valid and candidate not in parsed:
            parsed.append(candidate)
    return parsed


def _parse_textual_selection_metadata(text: str) -> dict[str, dict[str, Any]]:
    """Parse optional per-arm delta/hypothesis metadata from textual-gradient input."""
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    items: list[Any] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("arms", "selected_arms", "dimensions", "selected_dimensions"):
            value = data.get(key)
            if isinstance(value, list):
                items = value
                break
    metadata: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        arm = str(item.get("arm") or item.get("dimension") or item.get("name") or "").strip()
        if arm not in _active_arms():
            continue
        entry: dict[str, Any] = {}
        if "delta" in item:
            try:
                entry["delta"] = float(item["delta"])
            except Exception:
                pass
        reason = item.get("reason") or item.get("hypothesis") or item.get("rationale")
        if reason:
            entry["reason"] = str(reason)
        if entry:
            metadata[arm] = entry
    return metadata


def _normalize_textual_selection(
    requested_arms: list[str],
    candidate_arms: list[str],
    group_size: int,
) -> tuple[list[str], list[str]]:
    """Validate LLM-selected arms while preserving normal RecHarness safety constraints."""
    notes: list[str] = []
    candidate_set = set(candidate_arms)
    selected: list[str] = []
    jumping_arms = _active_jumping_dims()

    for arm in requested_arms:
        if arm not in candidate_set:
            notes.append(f"ignored unavailable arm: {arm}")
            continue
        if _arm_blocked_by_selection(arm, selected):
            notes.append(f"ignored mutex/duplicate arm: {arm}")
            continue
        selected.append(arm)
        if arm in jumping_arms:
            return [arm], notes
        if len(selected) >= group_size:
            break

    for arm in candidate_arms:
        if len(selected) >= group_size:
            break
        if arm in jumping_arms:
            continue
        if _arm_blocked_by_selection(arm, selected):
            continue
        selected.append(arm)
    return selected[:group_size], notes


# ---------------------------------------------------------------------------
# Ceiling estimation
# ---------------------------------------------------------------------------

def estimate_ceiling(score_history: list[float], current_score: float) -> float:
    """Estimate the basin ceiling from the recent score trend.

    Uses the slope of the last 3 best-so-far scores as a proxy. This avoids
    opening jumps simply because a later candidate underperformed the incumbent.
    Returns a ceiling estimate >= current_score.
    """
    if len(score_history) < 2:
        # Not enough history: assume moderate unexplored gap
        return current_score + 0.05

    best_so_far: list[float] = []
    running_best = float("-inf")
    for score in score_history:
        running_best = max(running_best, float(score))
        best_so_far.append(running_best)
    recent = best_so_far[-3:] if len(best_so_far) >= 3 else best_so_far
    n = len(recent)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(recent) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom < 1e-9:
        # All same: converged
        return current_score + 0.001
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, recent)) / denom

    eps = 1e-4
    fast_threshold = 0.005   # score / round

    if abs(slope) < eps:
        # Converged — basin is nearly exhausted
        return current_score + 0.001
    elif slope > fast_threshold:
        # Still rising fast — extrapolate 2 more rounds
        return current_score + slope * 2
    else:
        # Slow climb — moderate unexplored gap
        return current_score + 0.02


def _stagnant_rounds(score_history: list[float], current_score: float) -> int:
    """Count consecutive rounds without a best-score improvement."""
    if not score_history:
        return 0

    best_so_far: list[float] = []
    running_best = float("-inf")
    for score in score_history:
        running_best = max(running_best, float(score))
        best_so_far.append(running_best)

    if current_score > running_best:
        return 0

    stagnant = 0
    for idx in range(len(best_so_far) - 1, 0, -1):
        if best_so_far[idx] > best_so_far[idx - 1]:
            break
        stagnant += 1
    return stagnant


def _get_candidate_arms(
    state: ThompsonState,
    current_score: float,
    remaining_rounds: int,
) -> list[str]:
    """Decide whether to open basin-jumping this round."""
    jump_state = _jump_gate_state(state, current_score, remaining_rounds)
    if jump_state["jump_gate_open"]:
        return _active_arms()          # basin exhausted — open jumping
    return _active_exploiting_arms()


def _jump_gate_state(
    state: ThompsonState,
    current_score: float,
    remaining_rounds: int,
) -> dict[str, Any]:
    """Return basin-ceiling jump gate diagnostics for the current round."""
    ceiling = estimate_ceiling(state.score_history, current_score)
    gap = ceiling - current_score
    stagnant_rounds = _stagnant_rounds(state.score_history, current_score)
    min_jump_budget = 600.0 * (N_RETUNE + 1)
    active_pending_jumps = [
        entry for entry in state.pending_queue
        if state.round_idx - int(entry.get("t_start", state.round_idx)) < N_RETUNE
    ]
    provisional_active = bool(getattr(state, "provisional_incumbent", {}))

    suppressed_reasons: list[str] = []
    if active_pending_jumps:
        suppressed_reasons.append("pending_retune_active")
    if provisional_active:
        suppressed_reasons.append("provisional_incumbent_active")
    if gap >= THETA:
        suppressed_reasons.append("basin_gap_above_threshold")
    if stagnant_rounds < MIN_STAGNANT_ROUNDS_BEFORE_JUMP:
        suppressed_reasons.append("insufficient_stagnant_rounds")
    if remaining_rounds <= N_RETUNE:
        suppressed_reasons.append("insufficient_remaining_rounds")
    if state.global_budget < min_jump_budget:
        suppressed_reasons.append("insufficient_budget")

    return {
        "ceiling": ceiling,
        "basin_gap": gap,
        "theta": THETA,
        "stagnant_rounds": stagnant_rounds,
        "min_stagnant_rounds_before_jump": MIN_STAGNANT_ROUNDS_BEFORE_JUMP,
        "min_jump_budget": min_jump_budget,
        "active_pending_jumps": active_pending_jumps,
        "jump_eligible_strategy_arms": sorted(_active_jump_eligible_strategy_arms()),
        "jump_gate_open": not suppressed_reasons,
        "jump_suppressed_reason": ";".join(suppressed_reasons),
    }


def _arm_blocked_by_selection(arm: str, selected_arms: list[str]) -> bool:
    """Return True when arm conflicts with an already selected mutex partner."""
    if arm in selected_arms:
        return True
    for group in _active_mutex_groups():
        if arm in group and any(selected in group for selected in selected_arms):
            return True
    return False


def _fill_selection_diversity(
    state: ThompsonState,
    selected_arms: list[str],
    candidate_arms: list[str],
    group_size: int,
    is_jumping_round: bool,
) -> list[str]:
    """Back-fill exploiting rounds to keep parallel slots diverse and occupied."""
    unique_selected: list[str] = []
    for arm in selected_arms:
        if arm not in unique_selected:
            unique_selected.append(arm)
    if is_jumping_round or len(unique_selected) >= group_size:
        return unique_selected[:group_size]

    exploit_pool = [arm for arm in candidate_arms if arm in _active_exploiting_arms()]
    if not exploit_pool:
        exploit_pool = _active_exploiting_arms()
    exploit_pool = sorted(
        exploit_pool,
        key=lambda arm: state.success_rate(arm),
        reverse=True,
    )
    for arm in exploit_pool:
        if len(unique_selected) >= group_size:
            break
        if _arm_blocked_by_selection(arm, unique_selected):
            continue
        unique_selected.append(arm)
    return unique_selected


# ---------------------------------------------------------------------------
# Hypothesis builder
# ---------------------------------------------------------------------------

def _build_hypothesis(arm: str, state: ThompsonState, diag_blob: str) -> str:
    """Build a hypothesis string combining Thompson signal, memory, and diagnostics."""
    parts: list[str] = []

    # Thompson signal
    arm_s = state.arms.get(arm)
    if _is_textual_gradient_policy():
        parts.append("Textual-gradient routing: arm selected by LLM from global experiment memory; α/β not used for routing")
    elif arm_s is not None:
        sr = arm_s.alpha / (arm_s.alpha + arm_s.beta)
        total = arm_s.alpha + arm_s.beta - 2   # subtract priors
        if total > 0:
            if sr > 0.6:
                parts.append(f"Thompson: high success rate ({sr:.2f}, {int(total)} obs) — exploit")
            elif sr < 0.4:
                parts.append(f"Thompson: low success rate ({sr:.2f}, {int(total)} obs) — explore cautiously")
            else:
                parts.append(f"Thompson: uncertain ({sr:.2f}, {int(total)} obs) — explore")

    # Keep candidate hypotheses compact. Full Skill/Memory context is injected
    # separately into Claude Code via gagc_memory_context.
    if state.skill_notes and state.skill_notes != "Cold start: no prior knowledge.":
        parts.append("Memory available in state; agent should write a short arm-specific hypothesis")

    # Rejected dims buffer — warn if this arm recently failed
    recent_rejects = [
        r for r in state.rejected_dims_buffer
        if r.get("dim") == arm and state.round_idx - r.get("round_idx", 0) <= 5
    ]
    if recent_rejects:
        reasons = "; ".join(r.get("reason", "unknown") for r in recent_rejects[-2:])
        parts.append(f"Recently rejected: {reasons}")

    # Diagnostic patterns
    blob = diag_blob.lower()
    # Match against the base dims in this arm
    base_dims = _expand_arm(arm)
    for pattern, rule_dim, hint in _active_diag_rules():
        if rule_dim in base_dims and re.search(pattern, blob):
            parts.append(hint)

    if not parts:
        parts.append(f"No strong signal — default exploration of {arm}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Tool 1: propose_action_group
# ---------------------------------------------------------------------------

def propose_action_group(
    state_json: str,
    diagnostics_json: str = "{}",
    group_size: int = 4,
    explore_sigma: float = 0.3,
    textual_selected_arms_json: str = "",
) -> str:
    """Generate G candidate MutationSpecs using the active routing policy.

    Default policy is Thompson Sampling. For the textual-gradient ablation,
    set routing policy to "textual_gradient" and pass LLM-selected arms via
    textual_selected_arms_json.

    Args:
        state_json: JSON-serialised ThompsonState (or empty for cold start).
            The tool reads authoritative state from the internal store automatically.
        diagnostics_json: JSON of the previous best TrialResult. Pass "{}" on iteration 1.
        group_size: Maximum number of parallel trials (default 4).
        explore_sigma: Unused — kept for API backward compatibility.
        textual_selected_arms_json: Optional LLM-selected arms for TS-off ablation.
            Accepts JSON list, {"arms": [...]}, or comma/newline-separated arm names.

    Returns:
        JSON array of 1 or group_size candidate dicts with fields:
          dimension, delta, estimated_cost_secs, code_edits, code_diff, code_content,
          code_hint, hypothesis, is_jumping (bool).
        delta sign: positive = increase / forward variant, negative = decrease / conservative.
    """
    global _LAST_SELECTION_LOG, _LAST_PROPOSED_CANDIDATES

    # Read authoritative state
    raw = ""
    if _STORE is not None:
        try:
            item = _STORE.get(_STORE_NAMESPACE, _STORE_KEY)
            raw = item.value["content"] if item else ""
        except Exception:
            raw = ""
    if not raw:
        raw = state_json

    if raw and raw.strip() not in ("", "{}"):
        state = ThompsonState.from_json(raw)
    else:
        state = ThompsonState()

    # Ensure state.arms points at the current basin posterior.
    _sync_current_basin(state)

    if not getattr(state, "baseline_done", False):
        cost = max(_configured_trial_floor_secs(), min(3600.0, max(float(state.global_budget), 0.0)))
        candidates = [{
            "dimension": "baseline",
            "arm": "baseline",
            "composite_dims": ["baseline"],
            "delta": 0.0,
            "estimated_cost_secs": cost,
            "parallel_group_estimated_wall_secs": cost,
            "budget_safe_if_global_budget_gte": cost,
            "budget_accounting": "Parallel group cost is the sum of candidate estimated_cost_secs; the baseline is a single trial, so its group cost equals its one estimate.",
            "code_edits": [],
            "code_diff": "",
            "code_content": "",
            "code_hint": "Run the unmodified cold-start best.py once to establish the baseline score.",
            "hypothesis": "Mandatory cold-start baseline before any mutation.",
            "is_jumping": False,
            "is_baseline": True,
        }]
        _LAST_SELECTION_LOG = {
            "timestamp": _utc_now_iso(),
            "round_idx": state.round_idx,
            "benchmark_mode": _BENCHMARK_MODE,
            "routing_policy": "baseline_first",
            "group_size": 1,
            "selected_arms": ["baseline"],
            "selection_notes": ["baseline_done is false; returning mandatory no-op cold-start baseline"],
            "is_jumping_round": False,
            "reward_policy": "validation_only",
            "candidates": candidates,
        }
        _LAST_PROPOSED_CANDIDATES = deepcopy(candidates)
        _write_json_log(f"selection/round_{state.round_idx + 1:04d}.json", _LAST_SELECTION_LOG)
        return json.dumps(candidates, indent=2)

    required_group_size = _SERVER_CONFIG.logical_group_size
    if required_group_size is not None:
        if required_group_size < 2:
            raise RuntimeError(
                "logical_group_size must be at least 2 for normal Thompson groups; "
                "singleton groups have zero normalized advantage"
            )
        group_size = required_group_size

    diag: dict[str, Any] = json.loads(diagnostics_json) if diagnostics_json.strip() else {}
    diag_blob = " ".join([
        str(diag.get("error_message", "")),
        str(diag.get("stdout_tail", "")),
        str(diag.get("stderr_tail", "")),
        "out of memory" if diag.get("oom") else "",
        "timeout" if diag.get("timed_out") else "",
    ])

    # Determine current best score for ceiling estimation
    current_score = float(diag.get("val_score") or 0.0)
    if state.score_history:
        current_score = max(current_score, max(state.score_history))

    remaining_rounds = max(1, 24 - state.round_idx)
    jump_gate = _jump_gate_state(state, current_score, remaining_rounds)
    active_pending_jumps = list(jump_gate["active_pending_jumps"])
    if active_pending_jumps:
        candidate_arms = _active_exploiting_arms()
    else:
        candidate_arms = _active_arms() if jump_gate["jump_gate_open"] else _active_exploiting_arms()

    fixed_discovery_arm = _active_fixed_discovery_arm()
    if fixed_discovery_arm:
        candidate_arms = [arm for arm in candidate_arms if arm != fixed_discovery_arm]

    routing_policy = _routing_policy_enabled()
    selection_notes: list[str] = []
    requested_arms: list[str] = []
    textual_selection_metadata: dict[str, dict[str, Any]] = {}
    if _is_textual_gradient_policy():
        requested_arms = _parse_requested_arms(textual_selected_arms_json)
        textual_selection_metadata = _parse_textual_selection_metadata(textual_selected_arms_json)
        selected_arms, selection_notes = _normalize_textual_selection(
            requested_arms=requested_arms,
            candidate_arms=candidate_arms,
            group_size=group_size,
        )
        if not requested_arms:
            selection_notes.append("no textual_selected_arms_json provided; filled deterministic exploiting arms")
    else:
        selected_arms = thompson_sample(
            state=state,
            candidate_arms=candidate_arms,
            mutex_groups=_active_mutex_groups(),
            G=group_size,
        )

    if (
        fixed_discovery_arm
        and fixed_discovery_arm in _active_arms()
        and fixed_discovery_arm not in selected_arms
        and group_size > 1
    ):
        selected_arms = [fixed_discovery_arm] + [
            arm for arm in selected_arms
            if arm != fixed_discovery_arm
        ][: max(group_size - 1, 0)]
        selection_notes.append(
            f"fixed discovery slot inserted: {fixed_discovery_arm}; remaining slots use {routing_policy} routing"
        )

    # Detect jumping round. In this old-lowlevel-arm ablation, Amazon Reviews
    # uses explicit low-level jumping dimensions such as change_architecture and
    # change_loss_function; when selected, the round is isolated as G=1.
    jumping_arms = _active_jumping_dims()
    is_jumping_round = any(a in jumping_arms for a in selected_arms)
    jump_selected_arm = next((a for a in selected_arms if a in jumping_arms), None)
    if not jump_gate["jump_gate_open"] and jump_gate["jump_suppressed_reason"]:
        selection_notes.append(f"jump suppressed: {jump_gate['jump_suppressed_reason']}")

    if is_jumping_round:
        # Keep only the highest-ranked jumping arm (G=1)
        selected_arms = [jump_selected_arm or next(a for a in selected_arms if a in jumping_arms)]
    else:
        selected_arms = _fill_selection_diversity(
            state,
            selected_arms,
            candidate_arms,
            group_size,
            is_jumping_round,
        )

    hint_map = _active_dimension_hints()
    # global_budget is in compute-seconds: a parallel group's estimated cost is
    # the sum of its members' estimated_cost_secs (all equal to the trial floor),
    # i.e. len(selected_arms) * floor. This matches how executed groups are
    # charged (sum of per-member wall_time_secs) and handles G=1 jump rounds.
    group_estimated_wall_secs = len(selected_arms) * _configured_trial_floor_secs()

    candidates = []
    for arm in selected_arms:
        # Use alpha/beta ratio to determine direction (alpha > beta → positive delta)
        arm_s = state.arms.get(arm)
        if _is_textual_gradient_policy():
            direction = 1.0
        elif arm_s is not None:
            direction = 1.0 if arm_s.alpha >= arm_s.beta else -1.0
        else:
            direction = 1.0
        delta = direction * 1.0
        if _is_textual_gradient_policy() and arm in textual_selection_metadata:
            delta = float(textual_selection_metadata[arm].get("delta", delta))

        cost = _configured_trial_floor_secs()

        primary_dim = _expand_arm(arm)[0]
        hint = hint_map.get(arm, hint_map.get(primary_dim, f"Modify best.py to improve {arm}."))

        hyp = _build_hypothesis(arm, state, diag_blob)
        if _is_textual_gradient_policy() and arm in textual_selection_metadata:
            reason = textual_selection_metadata[arm].get("reason")
            if reason:
                hyp = hyp + f"; LLM routing rationale: {reason}"
        candidate = {
            "dimension": primary_dim,
            "arm": arm,
            "composite_dims": _expand_arm(arm),
            "strategy_arm": None,
            "round_role": "lowlevel_routing",
            "delta": delta,
            "estimated_cost_secs": cost,
            "parallel_group_estimated_wall_secs": group_estimated_wall_secs,
            "budget_safe_if_global_budget_gte": group_estimated_wall_secs,
            "budget_accounting": "Parallel group cost is sum(estimated_cost_secs) across the selected candidates.",
            "code_edits": [],
            "code_diff": "",
            "code_content": "",
            "code_hint": hint,
            "hypothesis": hyp,
            "is_jumping": bool(is_jumping_round and arm == selected_arms[0]),
        }
        candidate.update(_claude_code_backend_fields_for_arm(arm))
        candidates.append(candidate)

    selection_log = {
        "timestamp": _utc_now_iso(),
        "round_idx": state.round_idx,
        "benchmark_mode": _BENCHMARK_MODE,
        "routing_policy": routing_policy,
        "group_size": group_size,
        "remaining_rounds": remaining_rounds,
        "current_score": current_score,
        "ceiling": jump_gate["ceiling"],
        "basin_gap": jump_gate["basin_gap"],
        "theta": jump_gate["theta"],
        "stagnant_rounds": jump_gate["stagnant_rounds"],
        "min_stagnant_rounds_before_jump": jump_gate["min_stagnant_rounds_before_jump"],
        "jump_gate_open": jump_gate["jump_gate_open"],
        "jump_suppressed_reason": jump_gate["jump_suppressed_reason"],
        "jump_eligible_strategy_arms": jump_gate["jump_eligible_strategy_arms"],
        "current_basin_id": getattr(state, "current_basin_id", "default"),
        "basin_transfer_rho": _configured_basin_transfer_rho(),
        "active_pending_jumps": active_pending_jumps,
        "provisional_incumbent_active": bool(getattr(state, "provisional_incumbent", {})),
        "candidate_arms": candidate_arms,
        "requested_arms": requested_arms,
        "textual_selection_metadata": textual_selection_metadata,
        "selected_arms": selected_arms,
        "selection_notes": selection_notes,
        "parallel_group_estimated_wall_secs": group_estimated_wall_secs,
        "budget_safe_if_global_budget_gte": group_estimated_wall_secs,
        "budget_accounting": "Next group is budget-safe when global_budget >= parallel_group_estimated_wall_secs (= sum of candidate estimated_cost_secs).",
        "is_jumping_round": is_jumping_round,
        "reward_policy": "validation_only",
        "selection_metric": _SELECTION_METRIC,
        "arm_priors": {
            arm: {
                "alpha": state.get_or_create_arm(arm).alpha,
                "beta": state.get_or_create_arm(arm).beta,
                "success_rate": state.success_rate(arm),
            }
            for arm in selected_arms
        },
        "experiment_skill_hash": _hash_text(state.experiment_skill or ""),
        "recent_text_gradients": list(state.recent_text_gradients),
        "failure_memory": list(state.failure_memory),
        "candidates": candidates,
    }
    _LAST_SELECTION_LOG = selection_log
    _LAST_PROPOSED_CANDIDATES = deepcopy(candidates)
    _write_json_log(f"selection/round_{state.round_idx + 1:04d}.json", selection_log)

    return json.dumps(candidates, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: execute_trial
# ---------------------------------------------------------------------------


def _apply_edits(content: str, edits: list[dict]) -> tuple[str, str | None]:
    """Apply structured find/replace edits to file content."""
    for i, edit in enumerate(edits):
        if isinstance(edit, dict):
            find_str = edit.get("find", "")
            replace_str = edit.get("replace", "")
        elif isinstance(edit, (list, tuple)) and len(edit) == 2:
            find_str, replace_str = edit
        else:
            return content, f"edit[{i}]: expected {{'find': ..., 'replace': ...}} or [find, replace]"
        find_str = str(find_str)
        replace_str = str(replace_str)
        if not find_str:
            return content, f"edit[{i}]: 'find' is empty"

        if find_str in content:
            content = content.replace(find_str, replace_str, 1)
            continue

        def _norm(s: str) -> str:
            return "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip() for ln in s.splitlines())

        find_norm = _norm(find_str)
        content_lines = content.splitlines(keepends=True)
        norm_lines = [re.sub(r"[ \t]+", " ", ln.rstrip("\n\r")).rstrip() for ln in content_lines]
        find_lines = find_norm.splitlines()
        n_find = len(find_lines)

        matched_start = -1
        for j in range(len(norm_lines) - n_find + 1):
            if norm_lines[j: j + n_find] == find_lines:
                matched_start = j
                break

        if matched_start == -1:
            return content, (
                f"edit[{i}]: 'find' string not found in file (tried exact + whitespace-normalised).\n"
                f"  find={find_str!r}\n"
                f"  Tip: copy the exact lines from read_file output, preserving indentation."
            )

        replace_lines = replace_str.splitlines(keepends=True)
        if replace_lines and not replace_lines[-1].endswith("\n"):
            replace_lines[-1] += "\n"
        content_lines[matched_start: matched_start + n_find] = replace_lines
        content = "".join(content_lines)

    return content, None


def execute_trial(
    spec_json: str,
    script_path: str = "/workspace/best.py",
    timeout_multiplier: float = 1.5,
    slot_id: int = 0,
) -> str:
    """Apply a MutationSpec diff and run the training script on a dedicated GPU/CPU slot."""
    script_path = _resolve_script_path(script_path)
    spec_data = json.loads(spec_json)
    spec_data.setdefault("dimension", "tune_lr")
    spec_data.setdefault("delta", 0.0)
    requested_cost = float(spec_data.get("estimated_cost_secs") or 0.0)
    if _is_baseline_spec_data(spec_data):
        spec_data["estimated_cost_secs"] = max(requested_cost, 3600.0, _configured_trial_floor_secs())
    else:
        spec_data["estimated_cost_secs"] = max(requested_cost, _configured_trial_floor_secs())
    spec_data.setdefault("code_diff", "")
    spec = MutationSpec(**spec_data)
    hard_timeout = spec.estimated_cost_secs * timeout_multiplier
    if _BENCHMARK_MODE == "kuairec":
        hard_timeout = max(hard_timeout, 5400.0)

    if _uses_claude_code_backend(spec) and _has_inline_mutation_payload(spec):
        return _make_trial_result(
            spec, wall_time=0.0, timed_out=False, oom=False,
            val_score=_CRASH_SCORE, convergence_trace=[],
            error=(
                "claude_code backend specs must not include inline mutation payloads; "
                "leave code_edits/code_content/code_diff and predict edits empty"
            ),
        )

    if _is_canonical_workspace_script(script_path) and _spec_has_mutation(spec):
        return _make_trial_result(
            spec, wall_time=0.0, timed_out=False, oom=False,
            val_score=_CRASH_SCORE, convergence_trace=[],
            error=(
                "unsafe direct mutation refused: execute_trial only allows no-op baseline "
                "runs on canonical best.py; use execute_trial_group for mutations"
            ),
        )

    train_path, predict_path = _resolve_trial_files(script_path)

    if _BENCHMARK_MODE == "e2_shared":
        if _E2_RUNTIME is None:
            raise RuntimeError("E2 shared runtime is not configured")
        # The adapter returns RH wire fields from a trusted shared worker. An
        # external exception propagates to the existing pending-group recovery.
        return json.dumps(_E2_RUNTIME.execute(
            train_path, spec.to_dict(), baseline=_is_baseline_spec_data(spec.to_dict()),
            timeout_secs=min(hard_timeout, spec.extra.get("e1_training_timeout_secs", hard_timeout))),
            indent=2)

    if _BENCHMARK_MODE == "e1_ml1m":
        if _E1_RUNTIME is None:
            raise RuntimeError("E1 native runtime is not configured; legacy/Claude execution is forbidden")
        try:
            outcome = _E1_RUNTIME.execute(train_path, spec.to_dict(),
                baseline=_is_baseline_spec_data(spec.to_dict()),
                timeout_secs=min(hard_timeout, spec.extra.get("e1_training_timeout_secs", hard_timeout)))
            ok = outcome.get("status") == "ok"
            score = float(outcome["ndcg_at_10"]) if ok else _CRASH_SCORE
            if ok and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Invalid trusted native NDCG@10 result")
            trial = json.loads(_make_trial_result(spec, wall_time=float(outcome.get("wall_time_secs", 0)),
                timed_out=bool(outcome.get("timed_out")), oom=bool(outcome.get("oom")),
                val_score=score if ok else _CRASH_SCORE,
                convergence_trace=outcome.get("convergence_trace", [score] if ok else []),
                error=None if ok else outcome.get("error", "native training failed"),
                val_metrics={"NDCG@10": score, "Recall@10": outcome.get("recall_at_10")} if ok else None))
            trial["native_evidence"] = {key: outcome[key] for key in
                ("status", "source_sha256", "artifacts", "shared_baseline", "reused_completed_result")
                if key in outcome}
            return json.dumps(trial, indent=2)
        except Exception as exc:
            from gagc.e1_runtime import BudgetExhausted
            from openai import APIError
            from httpx import TransportError
            provider_failure = False
            cause = exc
            while cause is not None:
                if isinstance(cause, BudgetExhausted):
                    raise cause
                provider_failure |= isinstance(cause, (APIError, TransportError))
                cause = cause.__cause__
            if provider_failure:
                # An unavailable provider leaves this member pending. Never pass a
                # fabricated zero to promotion, Thompson updates, or Skill memory.
                raise
            trial = json.loads(_make_trial_result(spec, wall_time=0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=f"E1 implementation/runtime failure: {type(exc).__name__}: {exc}"))
            return json.dumps(trial, indent=2)

    if _uses_claude_code_backend(spec):
        try:
            from gagc.claude_code_backend import claude_backend_enabled, run_claude_code_trial
        except Exception as exc:
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=f"claude_code_backend import failed: {exc}",
            )
        if not claude_backend_enabled():
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error="claude_code_backend disabled; set GAGC_ENABLE_CLAUDE_BACKEND=1 to enable",
            )

        env, cpu_set = _trial_runtime_env(slot_id)
        outcome = run_claude_code_trial(
            trial_dir=os.path.dirname(train_path),
            train_path=train_path,
            predict_path=predict_path,
            spec_data=spec.to_dict(),
            env=env,
            hard_timeout=hard_timeout,
            benchmark_mode=_BENCHMARK_MODE,
            validation_callback=lambda: _validate_trial_syntax_and_contract(train_path, predict_path),
        )
        stdout = outcome.train_stdout
        stderr = outcome.train_stderr
        stdout_tail = stdout[-3000:] if len(stdout) > 3000 else stdout
        stderr_tail = stderr[-2000:] if len(stderr) > 2000 else stderr
        val_score = _CRASH_SCORE
        convergence_trace: list[float] = []
        val_metrics: dict | None = None
        if outcome.success:
            if _BENCHMARK_MODE == "kuairec":
                val_score, convergence_trace, val_metrics = _run_kuairec_eval(stdout)
            else:
                val_score, convergence_trace, val_metrics = _run_benchmark_eval(predict_path, stdout)
        claude_log_group = str(spec.extra.get("trial_group_id") or "unknown_group")
        claude_log_trial = str(spec.extra.get("trial_id") or slot_id)
        _write_json_log(
            f"trials/claude_code_{claude_log_group}_{claude_log_trial}.json",
            outcome.to_log_dict(),
        )
        return _make_trial_result(
            spec, wall_time=outcome.wall_time_secs,
            timed_out=outcome.timed_out, oom=outcome.oom,
            val_score=(
                val_score if outcome.success
                else _OOM_SCORE if outcome.oom
                else _TIMEOUT_SCORE if outcome.timed_out
                else _CRASH_SCORE
            ),
            convergence_trace=convergence_trace,
            error=None if outcome.success else outcome.error_message,
            val_metrics=val_metrics,
            stdout_tail=stdout_tail, stderr_tail=stderr_tail,
        )

    if spec.code_edits:
        err = _apply_text_mutation(train_path, spec.code_edits, "", "train.py")
        if err:
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=err,
            )
    elif spec.code_content.strip():
        err = _apply_text_mutation(train_path, [], spec.code_content, "train.py")
        if err:
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=err,
            )
    elif spec.code_diff.strip():
        workspace_dir = os.path.dirname(train_path)
        patch_result = subprocess.run(
            ["patch", "-p1", "--input=/dev/stdin", train_path],
            input=spec.code_diff.encode(), capture_output=True, timeout=10,
        )
        if patch_result.returncode != 0:
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=f"patch failed: {patch_result.stderr.decode()[:500]}",
            )

    predict_edits = spec.extra.get("predict_code_edits") or []
    predict_content = spec.extra.get("predict_code_content") or ""
    if predict_edits or str(predict_content).strip():
        err = _apply_text_mutation(predict_path, predict_edits, str(predict_content), "predict.py")
        if err:
            return _make_trial_result(
                spec, wall_time=0.0, timed_out=False, oom=False,
                val_score=_CRASH_SCORE, convergence_trace=[],
                error=err,
            )

    err = _validate_trial_syntax_and_contract(train_path, predict_path)
    if err:
        return _make_trial_result(
            spec, wall_time=0.0, timed_out=False, oom=False,
            val_score=_CRASH_SCORE, convergence_trace=[],
            error=err,
        )

    env, cpu_set = _trial_runtime_env(slot_id)

    start = time.time()
    timed_out = False
    oom = False
    val_score = _CRASH_SCORE
    convergence_trace: list[float] = []
    error_message: str | None = None
    val_metrics: dict | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def _set_affinity() -> None:
        try:
            os.sched_setaffinity(0, set(cpu_set))
        except (AttributeError, OSError):
            pass

    try:
        proc = subprocess.run(
            [sys.executable, train_path],
            capture_output=True, timeout=hard_timeout, env=env, preexec_fn=_set_affinity,
        )
        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")
        stdout_tail = stdout[-3000:] if len(stdout) > 3000 else stdout
        stderr_tail = stderr[-2000:] if len(stderr) > 2000 else stderr

        if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
            oom = True
            val_score = _OOM_SCORE
            error_message = "OOM"
        elif proc.returncode != 0:
            error_message = stderr[-2000:] if len(stderr) > 2000 else stderr
        else:
            if _BENCHMARK_MODE == "kuairec":
                bench_score, convergence_trace, val_metrics = \
                    _run_kuairec_eval(stdout)
            else:
                predict_script = predict_path
                bench_score, convergence_trace, val_metrics = \
                    _run_benchmark_eval(predict_script, stdout)
            val_score = bench_score

    except subprocess.TimeoutExpired:
        timed_out = True
        val_score = _TIMEOUT_SCORE
        error_message = "timeout"

    wall_time = time.time() - start

    return _make_trial_result(
        spec, wall_time=wall_time, timed_out=timed_out, oom=oom,
        val_score=val_score, val_metrics=val_metrics,
        convergence_trace=convergence_trace, error=error_message,
        stdout_tail=stdout_tail, stderr_tail=stderr_tail,
    )


def _run_kuairec_eval(
    stdout: str,
) -> tuple[float, list[float], dict | None]:
    convergence_trace = _parse_gr_convergence_trace(stdout)
    xauc = _CRASH_SCORE
    mae = float("inf")
    wr_xauc = None
    wr_mae = None
    for line in stdout.splitlines():
        m = re.match(r"^\s*XAUC\s*=\s*([0-9.]+)\s*$", line, re.IGNORECASE)
        if m:
            xauc = float(m.group(1))
        m = re.match(r"^\s*MAE\s*=\s*([0-9.]+)\s*$", line, re.IGNORECASE)
        if m:
            mae = float(m.group(1))
        m = re.match(r"^\s*WR_XAUC\s*=\s*([0-9.]+)\s*$", line, re.IGNORECASE)
        if m:
            wr_xauc = float(m.group(1))
        m = re.match(r"^\s*WR_MAE\s*=\s*([0-9.]+)\s*$", line, re.IGNORECASE)
        if m:
            wr_mae = float(m.group(1))
    if not convergence_trace and xauc != _CRASH_SCORE:
        convergence_trace = [xauc]
    val_metrics = {"WT-XAUC": xauc, "WT-MAE": mae} if xauc != _CRASH_SCORE else None
    if val_metrics is not None:
        if wr_xauc is not None:
            val_metrics["WR-XAUC"] = wr_xauc
        if wr_mae is not None:
            val_metrics["WR-MAE"] = wr_mae
    return xauc, convergence_trace, val_metrics


def _parse_gr_convergence_trace(stdout: str) -> list[float]:
    scores: list[float] = []
    for line in stdout.splitlines():
        m = re.search(r"WT_XAUC[=:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            scores.append(float(m.group(1)))
    return scores


def _run_benchmark_eval(
    predict_script: str, stdout: str
) -> tuple[float, list[float], dict | None]:
    convergence_trace = _parse_convergence_trace(stdout)
    if not os.path.isfile(predict_script):
        score, trace = _parse_metrics(stdout)
        return score, trace or convergence_trace, None

    val_hr10 = _CRASH_SCORE
    val_metrics: dict | None = None

    try:
        from gagc.benchmarks.amazon_reviews.harness import evaluate_val_fast
        from gagc.benchmarks.amazon_reviews.task import active_datasets
        val_result = evaluate_val_fast(
            predict_script=predict_script,
            data_dir=_DATA_DIR,
            max_eval_users=200,
            datasets=active_datasets(),
            verbose=False,
        )
        primary_metric = os.getenv("GAGC_PRIMARY_METRIC", "HR@10").strip() or "HR@10"
        val_hr10 = float(val_result.aggregate.get(primary_metric, val_result.primary_metric))
        val_metrics = {ds: r.metrics for ds, r in val_result.per_dataset.items()}
        if not convergence_trace:
            convergence_trace = [val_hr10]
    except Exception:
        score, trace = _parse_metrics(stdout)
        val_hr10 = score
        convergence_trace = trace or convergence_trace

    return val_hr10, convergence_trace, val_metrics


def _parse_convergence_trace(stdout: str) -> list[float]:
    scores: list[float] = []
    for line in stdout.splitlines():
        m = re.search(r"val[_\s]score[:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            scores.append(float(m.group(1)))
    return scores


def _parse_metrics(stdout: str) -> tuple[float, list[float]]:
    scores = _parse_convergence_trace(stdout)
    if scores:
        return scores[-1], scores
    return _CRASH_SCORE, []


def _make_trial_result(
    spec: MutationSpec,
    *,
    wall_time: float,
    timed_out: bool,
    oom: bool,
    val_score: float,
    convergence_trace: list[float],
    error: str | None,
    val_metrics: dict | None = None,
    test_score: float | None = None,
    test_metrics: dict | None = None,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> str:
    return json.dumps(
        {
            "spec": _spec_to_dict(spec),
            "wall_time_secs": wall_time,
            "timed_out": timed_out,
            "oom": oom,
            "val_score": val_score,
            "val_metrics": val_metrics,
            "test_score": test_score,
            "test_metrics": test_metrics,
            "convergence_trace": convergence_trace,
            "error_message": error,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Tool 3: update_thompson_state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ExperimentSkill helpers
# ---------------------------------------------------------------------------

_MAX_TEXT_GRADIENTS = 3     # compact rolling window for recent winner lessons
_MAX_FAILURE_MEMORY = 10    # cap on concrete avoid rules
_MAX_EXPERIMENT_MEMORY = 50 # compact audit window for experiment digests
_VAL_TEST_GAP_WARN = 0.025  # fast-val HR@10 can be noisy; flag large generalisation gaps
_MAX_AGENT_HYPOTHESIS_CHARS = 800
_MAX_AGENT_HYPOTHESES_JSON_CHARS = 6000
_MAX_CLAUDE_IMPLEMENTATION_PROMPT_CHARS = 1800


def _benchmark_score(result: dict) -> float:
    """Return the search/promotion reward; RecHarness search is validation-only."""
    return float(result.get("val_score", 0.0))


def _is_baseline_result(result: dict) -> bool:
    spec = result.get("spec") or {}
    return bool(
        spec.get("is_baseline")
        or (
            not spec.get("code_edits")
            and not spec.get("code_content")
            and not spec.get("code_diff")
            and float(spec.get("delta", 0.0) or 0.0) == 0.0
        )
    )


def _is_valid_trial(result: dict) -> bool:
    """Return True if the trial completed without timeout / OOM / crash."""
    if result.get("timed_out") or result.get("oom"):
        return False
    val = _metric_float(result.get("val_score"), _CRASH_SCORE)
    return val not in (_TIMEOUT_SCORE, _OOM_SCORE, _CRASH_SCORE)


def _metric_float(value: Any, default: float) -> float:
    """Return a finite-ish metric float, tolerating None / malformed values."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _has_test_score(result: dict) -> bool:
    return result.get("test_score") is not None


def _has_required_benchmark_score(result: dict) -> bool:
    """Return True when the search/promotion validation score is present."""
    return result.get("val_score") is not None


def _select_winner_by_test(results: list[dict]) -> dict | None:
    """Deprecated compatibility wrapper; search winner selection is validation-only."""
    return _select_winner_by_benchmark(results)


def _select_winner_by_final_test(results: list[dict]) -> dict | None:
    """Pick the valid final-eval result with highest test_score (ties: lower MAE)."""
    valid = [r for r in results if _is_valid_trial(r) and _has_test_score(r)]
    if not valid:
        return None

    def _key(r: dict):
        ts = _benchmark_score(r)
        mae = _metric_float((r.get("test_metrics") or {}).get("MAE"), float("inf"))
        return (ts, -mae)

    return max(valid, key=_key)


def _select_winner_by_benchmark(results: list[dict]) -> dict | None:
    """Pick the valid trial by validation score (ties: lower validation MAE)."""
    valid = [r for r in results if _is_valid_trial(r) and _has_required_benchmark_score(r)]
    if not valid:
        return None

    def _key(r: dict):
        score = _benchmark_score(r)
        mae = _metric_float((r.get("val_metrics") or {}).get("MAE"), float("inf"))
        return (score, -mae)

    return max(valid, key=_key)


def _previous_best_test(state: "ThompsonState") -> float | None:
    """Return the best validation score selected before this round."""
    if state.score_history:
        return max(state.score_history)
    return None


def _previous_winner_val(state: "ThompsonState") -> float | None:
    """Return the most recent previous winner val_score from experiment_memory."""
    for digest in reversed(state.experiment_memory):
        if digest.get("is_winner") and digest.get("valid") and digest.get("val_score") is not None:
            try:
                return float(digest["val_score"])
            except (TypeError, ValueError):
                return None
    return None


def _summarize_patch(spec: dict) -> str:
    """Return a short human-readable summary of what the patch changed."""
    arm = spec.get("arm") or spec.get("dimension", "unknown")
    if _BENCHMARK_MODE in {"e1_ml1m", "e2_shared"}:
        hypothesis = spec.get("agent_hypothesis") or spec.get("hypothesis")
        if hypothesis:
            return f"{arm} hypothesis: {str(hypothesis)[:400]}"
    dim = spec.get("dimension", arm)
    delta = spec.get("delta", 0.0)
    direction = "↑" if float(delta) > 0 else "↓"
    edits = spec.get("code_edits") or []
    if edits:
        first = edits[0]
        if isinstance(first, dict):
            find_snip = str(first.get("find", ""))[:60].replace("\n", "⏎")
            repl_snip = str(first.get("replace", ""))[:60].replace("\n", "⏎")
        elif isinstance(first, (list, tuple)) and len(first) == 2:
            find_snip = str(first[0])[:60].replace("\n", "⏎")
            repl_snip = str(first[1])[:60].replace("\n", "⏎")
        else:
            find_snip = "<invalid edit>"
            repl_snip = "<invalid edit>"
        return f"{arm}/{dim} {direction}: `{find_snip}` → `{repl_snip}`"
    if spec.get("code_content"):
        return f"{arm}/{dim}: full file rewrite"
    return f"{arm}/{dim} {direction}"


def _normalize_patch_pattern(patch: str) -> str:
    """Normalize patch summaries for stable deduplication."""
    compact = re.sub(r"\s+", " ", str(patch)).strip().lower()
    compact = compact.replace("`", "")
    return compact[:120]


def _specific_patch_pattern(patch: str) -> str:
    """Extract the most useful short patch pattern from a verbose patch summary."""
    text = str(patch)
    search_texts = [text]
    if "→" in text:
        search_texts.insert(0, text.split("→", 1)[-1])
    patterns = [
        r"(?:batch|batch_size|BATCH_SIZE|GR_BATCH_SIZE)\s*[=:]\s*\d+",
        r"(?:embedding|embedding_dim|EMBEDDING_DIM|GR_HIDDEN_DIM|hidden_dim)\s*[=:]\s*\d+",
        r"(?:dropout|DROPOUT|GR_DROPOUT)\s*[=:]\s*[0-9.]+",
        r"(?:lr|LR|learning_rate|GR_LR)\s*[=:]\s*[0-9.eE+-]+",
        r"(?:num_layers|layers|N_LAYERS|GR_DEC_LAYERS)\s*[=:]\s*\d+",
        r"(?:num_heads|heads|N_HEAD|GR_N_HEAD)\s*[=:]\s*\d+",
    ]
    for candidate_text in search_texts:
        for pattern in patterns:
            match = re.search(pattern, candidate_text)
            if match:
                return match.group(0)
    if "→" in text:
        return text.split("→", 1)[-1].strip(" `")[:80]
    return text[:80]


def _val_support(current_val: float | None, prev_val: float | None) -> str:
    """Classify val_score movement relative to previous winner."""
    if current_val is None or prev_val is None:
        return "unavailable"
    diff = current_val - prev_val
    if diff > 0.001:
        return "positive"
    if diff < -0.001:
        return "negative"
    return "neutral"


def _val_test_gap_status(val_score: float | None, test_score: float | None) -> str:
    """Classify whether fast validation materially overstates benchmark test."""
    if val_score is None or test_score is None:
        return "unavailable"
    gap = val_score - test_score
    if gap > _VAL_TEST_GAP_WARN:
        return "overfit_or_noisy_val"
    if gap < -_VAL_TEST_GAP_WARN:
        return "test_exceeds_val"
    return "aligned"


def _dataset_hr10_values(result: dict) -> list[float]:
    metrics = result.get("val_metrics") or {}
    values: list[float] = []
    if not isinstance(metrics, dict):
        return values
    for item in metrics.values():
        if isinstance(item, dict) and item.get("HR@10") is not None:
            try:
                values.append(float(item["HR@10"]))
            except (TypeError, ValueError):
                pass
    return values


def _dataset_balance_status(result: dict) -> str:
    values = _dataset_hr10_values(result)
    if len(values) < 2:
        return "unavailable"
    spread = max(values) - min(values)
    if spread > 0.22:
        return "imbalanced"
    return "balanced"


def _classify_failure(result: dict) -> str:
    """Return a short failure type label."""
    if result.get("failure_stage") == "provider":
        return "provider_failure"
    if result.get("timed_out"):
        return "timeout"
    if result.get("oom"):
        return "oom"
    err = str(result.get("error_message") or "").lower()
    if "edit failed" in err or "not found in file" in err:
        return "edit_failed"
    if float(result.get("val_score", 0.0)) == _CRASH_SCORE:
        return "crash"
    return "implementation_failure"


def _build_experiment_digest(
    result: dict,
    state: "ThompsonState",
    is_winner: bool,
    prev_best: float | None,
    prev_winner_val: float | None,
) -> dict:
    """Compact summary of one trial result for experiment_memory."""
    spec = result.get("spec") or {}
    bench = _benchmark_score(result)
    val = float(result.get("val_score", 0.0))
    test = result.get("test_score")
    test_float = float(test) if test is not None else None
    val_metrics = result.get("val_metrics") or {}
    mae = float(val_metrics.get("MAE", val_metrics.get("WT-MAE", float("inf"))))
    valid = _is_valid_trial(result)
    return {
        "round_idx": state.round_idx,
        "arm": spec.get("arm") or spec.get("dimension", "unknown"),
        "patch_summary": _summarize_patch(spec),
        "benchmark_score": bench,
        "val_score": val,
        "test_score": test,
        "val_test_gap": None,
        "val_test_gap_status": "held_out_test_not_used",
        "dataset_balance_status": _dataset_balance_status(result),
        "mae": mae if mae != float("inf") else None,
        "is_winner": is_winner,
        "valid": valid,
        "failure_type": None if valid else _classify_failure(result),
        "delta_vs_prev_best": round(bench - prev_best, 5) if (
            prev_best is not None and result.get("failure_stage") != "provider"
        ) else None,
        "val_support": _val_support(val if valid else None, prev_winner_val),
    }


def _build_winner_lesson(digest: dict) -> str:
    """Build a one-sentence winner patch lesson for recent_text_gradients."""
    arm = digest["arm"]
    patch = digest["patch_summary"]
    delta = digest.get("delta_vs_prev_best")
    val_sup = digest.get("val_support", "unavailable")
    gap_status = digest.get("val_test_gap_status", "unavailable")
    gap = digest.get("val_test_gap")
    balance_status = digest.get("dataset_balance_status", "unavailable")
    round_n = digest["round_idx"]

    delta_str = f"+{delta:.4f}" if delta is not None and delta >= 0 else (f"{delta:.4f}" if delta is not None else "unknown")
    if delta is None:
        lesson = f"Round {round_n} winner established the benchmark reference using {patch}."
    else:
        verb = "improved" if delta >= 0 else "changed"
        lesson = f"Round {round_n} winner {verb} validation score by {delta_str} using {patch}."
    if val_sup == "negative":
        lesson += (
            " Val support was negative; if this direction is selected again, preserve the useful validation signal "
            "but make smaller follow-up edits or change only one nearby factor."
        )
    elif val_sup == "positive":
        lesson += " Val score also improved — high confidence direction."
    elif val_sup == "neutral":
        lesson += " Val score was neutral; keep follow-up edits conservative."
    if gap_status == "overfit_or_noisy_val":
        lesson += f" Fast val exceeded held-out test by {gap:.4f}; treat val as noisy."
    elif gap_status == "test_exceeds_val":
        lesson += f" Test exceeded fast val by {-gap:.4f}; do not discard this direction solely on val."
    if balance_status == "imbalanced":
        lesson += " Per-dataset HR@10 was imbalanced; prefer follow-ups that preserve weaker datasets, not only aggregate gain."
    return lesson


def _extract_failure_avoid(digest: dict) -> dict | None:
    """Extract a concrete avoid rule from a failed trial digest."""
    if digest.get("valid") or not digest.get("failure_type"):
        return None
    patch = digest.get("patch_summary", "")
    pattern = _specific_patch_pattern(patch)
    arm = digest.get("arm", "unknown")
    failure_type = digest["failure_type"]

    if failure_type == "provider_failure":
        avoid = (
            f"Provider request failed while attempting `{pattern}`; no development "
            "metric was produced. This does not refute the mechanism or justify "
            "avoiding this patch on scientific grounds."
        )
    elif failure_type == "oom":
        avoid = f"Do not repeat `{pattern}` — caused OOM. Reduce capacity or add gradient accumulation."
    elif failure_type == "timeout":
        avoid = f"Do not repeat `{pattern}` — caused timeout. Reduce model size or batch size."
    elif failure_type == "edit_failed":
        avoid = f"Patch pattern `{pattern}` failed to apply. Verify indentation and exact find/replace text before retrying."
    else:
        avoid = f"Patch pattern `{pattern}` failed with {failure_type}. Avoid repeating this exact change."

    return {
        "round_idx": digest["round_idx"],
        "arm": arm,
        "failure_type": failure_type,
        "patch_pattern": pattern,
        "normalized_pattern": _normalize_patch_pattern(pattern),
        "avoid_rule": avoid,
        "count": 1,
    }


def _merge_failure_memory(memory: list[dict], new_avoids: list[dict]) -> list[dict]:
    """Merge new avoid rules into memory, incrementing count for duplicates."""
    for new in new_avoids:
        matched = False
        for existing in memory:
            if (existing.get("arm") == new.get("arm")
                    and existing.get("failure_type") == new.get("failure_type")
                    and existing.get("normalized_pattern") == new.get("normalized_pattern")):
                existing["count"] = existing.get("count", 1) + 1
                existing["round_idx"] = new.get("round_idx", existing.get("round_idx", 0))
                existing["avoid_rule"] = new.get("avoid_rule", existing.get("avoid_rule", ""))
                matched = True
                break
        if not matched:
            memory.append(new)

    # Keep most recent, bounded
    if len(memory) > _MAX_FAILURE_MEMORY:
        memory.sort(key=lambda x: (x.get("count", 1), x.get("round_idx", 0)))
        memory = memory[-_MAX_FAILURE_MEMORY:]

    return memory


def _render_experiment_skill(state: "ThompsonState") -> str:
    """Render the current ExperimentSkill document from state."""
    lines = [
        "# RecHarness Experiment Skill",
        "",
        "## Objective",
        "Use benchmark val_score as the search and winner criterion.",
        "Do not use held-out test metrics during search.",
        (
            "Textual-gradient routing chooses arms from experiment memory; this skill guides both arm selection and code edits without using α/β ranking."
            if _is_textual_gradient_policy()
            else "Thompson Sampling chooses arms; this skill only guides how to edit code within selected arms."
        ),
        "",
    ]

    recent = state.recent_text_gradients[-3:] if state.recent_text_gradients else []
    lines.append("## Recent Patch Lessons")
    if recent:
        for lesson in recent:
            lines.append(f"- {lesson}")
    else:
        lines.append("No learned patch lessons yet.")
    lines.append("")

    lines.append("## Concrete Avoids")
    if state.failure_memory:
        for avoid in state.failure_memory[-10:]:
            lines.append(f"- {avoid.get('avoid_rule', '')}")
    else:
        lines.append("No concrete patch-level avoids yet.")
    lines.append("")

    return "\n".join(lines)


def _build_skill_guidance_for_arm(arm: str, state: "ThompsonState") -> str:
    """Return the relevant ExperimentSkill section appended as guidance for a specific arm."""
    parts = []

    # Include recent lessons that explicitly mention this arm name
    relevant_lessons = [
        g for g in state.recent_text_gradients
        if f" {arm}" in g or g.startswith(arm) or f"`{arm}" in g
    ]
    if relevant_lessons:
        parts.append(f"ExperimentSkill lessons for {arm}: " + " | ".join(relevant_lessons[-2:]))

    # Include arm-specific avoids
    arm_avoids = [
        m["avoid_rule"] for m in state.failure_memory
        if m.get("arm") == arm
    ]
    if arm_avoids:
        parts.append(f"Avoids for {arm}: " + "; ".join(arm_avoids[-3:]))

    if not parts:
        recent = state.recent_text_gradients[-_MAX_TEXT_GRADIENTS:]
        if recent:
            parts.append("Recent ExperimentSkill lessons: " + " | ".join(recent))
        global_avoids = [m.get("avoid_rule", "") for m in state.failure_memory[-3:] if m.get("avoid_rule")]
        if global_avoids:
            parts.append("Concrete avoids: " + "; ".join(global_avoids))

    return "; ".join(parts) if parts else ""



def update_thompson_state(
    trial_results_json: str,
    state_json: str = "",
    trial_group_id: str = "",
) -> str:
    """Update Thompson Sampling α/β for all trials and persist SkillOpt memory.

    Reads authoritative ThompsonState from the internal store, updates α/β using
    standard independent Thompson update (each dim gets +1 to α or β), processes
    the pending_queue for delayed basin-jumping back-fill, appends to score_history,
    records failures in rejected_dims_buffer, and writes back to the store.

    Args:
        trial_results_json: JSON array of TrialResult dicts, or "__LAST_RESULTS__" to
            read the latest raw execute_trial_group payload from durable/cache storage.
        state_json: Ignored — kept for backward compatibility only.
        trial_group_id: Optional trial_group_id from execute_trial_group results. When
            provided, durable fallback results with a different group_id are rejected.
            Also used to match _LAST_PROMOTION_RESULT for gating score_history writes.

    Returns:
        Updated ThompsonState JSON.
    """
    # Read authoritative state
    if _STORE is not None:
        try:
            item = _STORE.get(_STORE_NAMESPACE, _STORE_KEY)
            raw = item.value["content"] if item else ""
        except Exception:
            raw = state_json
    else:
        raw = state_json

    if raw:
        state = ThompsonState.from_json(raw)
    else:
        state = ThompsonState()

    raw_results_arg = (trial_results_json or "").strip()
    if (
        _STRICT_VALIDATION_ONLY
        and raw_results_arg not in {"", "__LAST_RESULTS__", "LAST_RESULTS"}
    ):
        _write_json_log("state/rejected_raw_results_update.json", {
            "timestamp": _utc_now_iso(),
            "trial_group_id": trial_group_id,
            "reason": "search-only state update accepts only the latest cached trial group",
            "state_unchanged": True,
        })
        return state.to_json()

    results_data, used_last_raw_results = _normalise_trial_results_input(
        trial_results_json,
        expected_group_id=trial_group_id,
        allow_durable_fallback=not _STRICT_VALIDATION_ONLY,
    )
    if _STRICT_VALIDATION_ONLY:
        strict_error = _strict_group_results_error(results_data)
        if strict_error:
            _write_json_log("state/rejected_search_group_update.json", {
                "timestamp": _utc_now_iso(),
                "trial_group_id": trial_group_id,
                "reason": strict_error,
                "state_unchanged": True,
            })
            return state.to_json()

    _sync_current_basin(state)

    state_before_log = _state_summary(state)
    budget_before = state.global_budget
    arms_before = {
        name: {"alpha": arm.alpha, "beta": arm.beta}
        for name, arm in state.arms.items()
    }
    if results_data and all(r.get("tool_error") or (r.get("spec") or {}).get("is_tool_error") for r in results_data):
        _write_json_log("state/rejected_tool_error_update.json", {
            "timestamp": _utc_now_iso(),
            "trial_group_id": trial_group_id,
            "reason": "trial results contain only tool errors",
            "results": results_data,
            "state_unchanged": True,
        })
        return state.to_json()
    if trial_group_id and not results_data:
        _write_json_log(f"state/rejected_update_{trial_group_id}.json", {
            "timestamp": _utc_now_iso(),
            "trial_group_id": trial_group_id,
            "reason": "no matching full trial results for requested trial_group_id",
            "state_unchanged": True,
        })
        return state.to_json()
    effective_group_id = trial_group_id or (
        results_data[0].get("trial_group_id", "") if results_data else ""
    )
    if effective_group_id and effective_group_id in getattr(state, "processed_trial_group_ids", []):
        _write_json_log(f"state/duplicate_update_{effective_group_id}.json", {
            "timestamp": _utc_now_iso(),
            "trial_group_id": effective_group_id,
            "reason": "trial_group_id already processed",
            "state_unchanged": True,
        })
        return state.to_json()
    is_baseline_round = bool(results_data) and all(_is_baseline_result(r) for r in results_data)
    _promo = _LAST_PROMOTION_RESULT
    _promo_covers_this_group = bool(
        _promo
        and effective_group_id
        and _promo.get("trial_group_id") == effective_group_id
    )
    if is_baseline_round:
        _promotion_succeeded = True
        _kept_incumbent = False
        _provisional_jump = False
        state.baseline_done = True
    elif _promo_covers_this_group:
        _promotion_succeeded = bool(_promo.get("promoted"))
        _kept_incumbent = bool(_promo.get("kept_incumbent"))
        _provisional_jump = bool(_promo.get("provisional_jump"))
    elif effective_group_id:
        _promotion_succeeded = False
        _kept_incumbent = False
        _provisional_jump = False
    else:
        _promotion_succeeded = True
        _kept_incumbent = False
        _provisional_jump = False

    _winner_for_promotion = _select_winner_by_benchmark(results_data)
    if (not is_baseline_round
            and effective_group_id
            and _winner_for_promotion is not None
            and not _promotion_succeeded
            and not _kept_incumbent):
        _write_json_log(f"state/unpromoted_update_{effective_group_id}.json", {
            "timestamp": _utc_now_iso(),
            "trial_group_id": effective_group_id,
            "reason": "valid winner exists but matching promote_winner success was not observed",
            "promotion_status": deepcopy(_promo),
            "state_unchanged": True,
        })
        return state.to_json()
    rewards = [_benchmark_score(r) for r in results_data]
    advantages = compute_group_advantages(rewards)

    # Baselines must be captured before this round mutates score_history / memory.
    prev_best = _previous_best_test(state)
    prev_winner_val = _previous_winner_val(state)
    if _provisional_jump:
        snapshot = deepcopy(_promo.get("previous_incumbent_snapshot") or {})
        if snapshot:
            snapshot["t_start"] = state.round_idx + 1
            snapshot["dim"] = (_promo.get("spec") or {}).get("arm") or (_promo.get("spec") or {}).get("dimension", "")
            state.provisional_incumbent = snapshot

    # Advance round counter
    state.round_idx += 1

    # --- Standard Thompson updates for this round's trials ---
    arm_updates: list[dict[str, Any]] = []
    textual_gradient_mode = _is_textual_gradient_policy()
    for result, advantage in zip(results_data, advantages):
        spec_block = result.get("spec") or result
        dim = spec_block.get("dimension", "")
        arm = spec_block.get("arm", dim)   # prefer arm name if present
        jumping_dims = _active_jumping_dims()
        is_jumping = bool(spec_block.get("is_jumping")) or arm in jumping_dims or dim in jumping_dims

        val_score = float(result.get("val_score", 0.0))
        missing_required_score = not _has_required_benchmark_score(result)

        if is_baseline_round:
            arm_updates.append({
                "arm": arm,
                "dimension": dim,
                "advantage": advantage,
                "reward_used": _benchmark_score(result),
                "update": "baseline: no alpha/beta update",
            })
            continue

        if textual_gradient_mode:
            target_arm = arm if arm in state.arms else dim
            arm_s = state.get_or_create_arm(target_arm)
            reward = _benchmark_score(result)
            arm_updates.append({
                "arm": target_arm,
                "dimension": dim,
                "advantage": advantage,
                "reward_used": reward,
                "update": "textual_gradient: no alpha/beta update",
                "before": {"alpha": arm_s.alpha, "beta": arm_s.beta},
                "after": {"alpha": arm_s.alpha, "beta": arm_s.beta},
            })
            if val_score in (_TIMEOUT_SCORE, _OOM_SCORE, _CRASH_SCORE):
                reason = result.get("error_message") or ("OOM" if result.get("oom") else
                         "timeout" if result.get("timed_out") else "crash")
                state.rejected_dims_buffer.append({
                    "dim": target_arm,
                    "round_idx": state.round_idx,
                    "reason": str(reason)[:120],
                })
                if len(state.rejected_dims_buffer) > 50:
                    state.rejected_dims_buffer = state.rejected_dims_buffer[-50:]
            continue

        if is_jumping:
            target_arm = arm if arm in state.arms else dim
            arm_s = state.get_or_create_arm(target_arm)
            before_alpha, before_beta = arm_s.alpha, arm_s.beta
            if _is_valid_trial(result) and not missing_required_score:
                basin_switch = _switch_to_new_basin(state, target_arm)
                state.pending_queue.append({
                    "dim": target_arm,
                    "t_start": state.round_idx,
                    "base_score": prev_best,
                    "source_basin_id": basin_switch["previous_basin_id"],
                    "trial_basin_id": basin_switch["new_basin_id"],
                    "rho": basin_switch["rho"],
                })
                update_op = f"pending_queue += 1; basin -> {basin_switch['new_basin_id']}"
                _write_json_log(f"state/basin_switch_{state.round_idx:04d}_{target_arm}.json", {
                    "timestamp": _utc_now_iso(),
                    "dim": target_arm,
                    **basin_switch,
                })
            else:
                arm_s.beta += 1.0
                update_op = "beta += 1 (failed jumping; no pending)"
                reason = result.get("error_message") or (
                    "OOM" if result.get("oom") else "timeout" if result.get("timed_out") else "crash"
                )
                state.rejected_dims_buffer.append({
                    "dim": target_arm,
                    "round_idx": state.round_idx,
                    "reason": str(reason)[:120],
                })
                state.rejected_dims_buffer = state.rejected_dims_buffer[-50:]
            arm_updates.append({
                "arm": target_arm,
                "dimension": dim,
                "advantage": advantage,
                "reward_used": _benchmark_score(result),
                "update": update_op,
                "before": {"alpha": before_alpha, "beta": before_beta},
                "after": {"alpha": arm_s.alpha, "beta": arm_s.beta},
            })
            continue

        # Exploiting dim: standard independent Thompson update
        # advantage > 0 means this trial beat the group average → positive signal
        target_arm = arm if arm in state.arms else dim
        arm_s = state.get_or_create_arm(target_arm)
        before_alpha, before_beta = arm_s.alpha, arm_s.beta
        reward = _benchmark_score(result)
        below_incumbent = prev_best is not None and reward < prev_best
        noisy_val_gap = False
        valid_trial = _is_valid_trial(result)
        if valid_trial and not missing_required_score and advantage > 0 and not below_incumbent and not noisy_val_gap:
            arm_s.alpha += 1.0
            update_op = "alpha += 1"
        else:
            arm_s.beta += 1.0
            if not valid_trial:
                update_op = "beta += 1 (invalid trial)"
            elif missing_required_score:
                update_op = "beta += 1 (missing benchmark score)"
            elif below_incumbent:
                update_op = "beta += 1 (below incumbent)"
            elif noisy_val_gap:
                update_op = "beta += 1 (val/test gap)"
            else:
                update_op = "beta += 1"
        arm_updates.append({
            "arm": target_arm,
            "dimension": dim,
            "advantage": advantage,
            "reward_used": reward,
            "update": update_op,
            "before": {"alpha": before_alpha, "beta": before_beta},
            "after": {"alpha": arm_s.alpha, "beta": arm_s.beta},
        })

        # Also update component dims of composite arms so they reflect reality
        if target_arm in _active_composite_arms():
            for component_dim in _active_composite_arms()[target_arm]:
                comp_s = state.get_or_create_arm(component_dim)
                if valid_trial and not missing_required_score and advantage > 0 and not below_incumbent and not noisy_val_gap:
                    comp_s.alpha += 1.0
                else:
                    comp_s.beta += 1.0

        # Record failures in rejected_dims_buffer (SkillOpt rejected-edit buffer)
        if val_score in (_TIMEOUT_SCORE, _OOM_SCORE, _CRASH_SCORE):
            reason = result.get("error_message") or ("OOM" if result.get("oom") else
                     "timeout" if result.get("timed_out") else "crash")
            state.rejected_dims_buffer.append({
                "dim": target_arm,
                "round_idx": state.round_idx,
                "reason": str(reason)[:120],
            })
            # Keep buffer bounded
            if len(state.rejected_dims_buffer) > 50:
                state.rejected_dims_buffer = state.rejected_dims_buffer[-50:]

    # --- Determine promotion status for this round ---
    # If promote_winner was called for this same trial group and succeeded, allow
    # score_history and winner lessons to be written. If it failed or has not been
    # called yet for a grouped mutation run, skip score_history / winner lesson to
    # keep state consistent with disk state. Baseline rounds are exempt because they
    # intentionally do not promote code.
    _effective_group_id = effective_group_id
    # --- Update score_history (per-round best validation score) ---
    valid_scores = [
        _benchmark_score(r)
        for r in results_data
        if _is_valid_trial(r) and _has_required_benchmark_score(r)
    ]
    if valid_scores and _promotion_succeeded and not _provisional_jump:
        round_best = max(valid_scores)
        state.score_history.append(max(round_best, prev_best) if prev_best is not None else round_best)
        snapshot = getattr(state, "provisional_incumbent", {}) or {}
        if snapshot and prev_best is not None and round_best > prev_best:
            state.provisional_incumbent = {}
    elif valid_scores and _promotion_succeeded and _provisional_jump and prev_best is not None:
        state.score_history.append(prev_best)
    elif valid_scores and _kept_incumbent and prev_best is not None:
        state.score_history.append(prev_best)

    # --- Process pending basin-jumping back-fills after this round's score is recorded ---
    completed_pending = []
    for entry in state.pending_queue:
        dim = entry["dim"]
        t_start = int(entry["t_start"])
        if state.round_idx - t_start < N_RETUNE:
            continue

        start_idx = max(0, len(state.score_history) - N_RETUNE)
        slice_scores = state.score_history[start_idx:]
        base_score_raw = entry.get("base_score")
        base_score = float(base_score_raw) if base_score_raw is not None else (slice_scores[0] if slice_scores else 0.0)
        retune_best = max(slice_scores) if slice_scores else base_score
        cumulative_delta = retune_best - base_score

        arm_s = state.get_or_create_arm(dim)
        snapshot = getattr(state, "provisional_incumbent", {}) or {}
        basin_switch: dict[str, Any] | None = None
        if textual_gradient_mode:
            if cumulative_delta > 0 and snapshot and snapshot.get("dim") == dim:
                state.provisional_incumbent = {}
        elif cumulative_delta > 0:
            target_basin_id = entry.get("source_basin_id") or state.current_basin_id
            target_arms = state.basin_arms.get(target_basin_id, state.arms)
            target_arm = target_arms.get(dim)
            if target_arm is None:
                target_arm = ArmState(name=dim)
                target_arms[dim] = target_arm
            target_arm.alpha += 1.0
            state.basin_arms[target_basin_id] = target_arms
            if snapshot and snapshot.get("dim") == dim:
                state.provisional_incumbent = {}
        else:
            source_basin_id = entry.get("source_basin_id") or state.current_basin_id
            source_arms = state.basin_arms.get(source_basin_id, state.arms)
            source_arm = source_arms.get(dim)
            if source_arm is None:
                source_arm = ArmState(name=dim)
                source_arms[dim] = source_arm
            source_arm.beta += 1.0
            state.basin_arms[source_basin_id] = source_arms
            if snapshot and snapshot.get("dim") == dim:
                restore_err = _restore_provisional_incumbent(snapshot)
                _write_json_log(f"promotion/rollback_{state.round_idx:04d}_{dim}.json", {
                    "timestamp": _utc_now_iso(),
                    "dim": dim,
                    "restored": restore_err is None,
                    "error": restore_err,
                    "snapshot_score": snapshot.get("score"),
                    "base_score": base_score,
                    "retune_best": retune_best,
                    "cumulative_delta": cumulative_delta,
                })
                if restore_err is None:
                    state.provisional_incumbent = {}
            if entry.get("trial_basin_id") == state.current_basin_id:
                _switch_to_existing_basin(state, source_basin_id)
        completed_pending.append(entry)

    for entry in completed_pending:
        state.pending_queue.remove(entry)

    # --- ExperimentSkill update ---
    winner = _select_winner_by_benchmark(results_data)
    new_avoids: list[dict] = []
    new_winner_lesson: str | None = None
    digests: list[dict] = []
    for result in results_data:
        is_win = winner is not None and result is winner
        digest = _build_experiment_digest(result, state, is_win, prev_best, prev_winner_val)
        digests.append(digest)
        state.experiment_memory.append(digest)
        if is_win and _promotion_succeeded and not _provisional_jump:
            lesson = _build_winner_lesson(digest)
            new_winner_lesson = lesson
            state.recent_text_gradients.append(lesson)
            if len(state.recent_text_gradients) > _MAX_TEXT_GRADIENTS:
                state.recent_text_gradients = state.recent_text_gradients[-_MAX_TEXT_GRADIENTS:]
        elif is_win and _provisional_jump:
            lesson = (
                f"Round {state.round_idx} provisional jump entered a new basin using {digest.get('patch_summary')}. "
                "Do not treat it as a winner until retune beats the saved incumbent validation score."
            )
            state.recent_text_gradients.append(lesson)
            if len(state.recent_text_gradients) > _MAX_TEXT_GRADIENTS:
                state.recent_text_gradients = state.recent_text_gradients[-_MAX_TEXT_GRADIENTS:]
        elif not is_win:
            avoid = _extract_failure_avoid(digest)
            if avoid is not None:
                new_avoids.append(avoid)
    if new_avoids:
        state.failure_memory = _merge_failure_memory(state.failure_memory, new_avoids)
    # Keep experiment_memory bounded
    if len(state.experiment_memory) > _MAX_EXPERIMENT_MEMORY:
        state.experiment_memory = state.experiment_memory[-_MAX_EXPERIMENT_MEMORY:]
    state.experiment_skill = _render_experiment_skill(state)

    group_wall_time, budget_spend_method = _trial_group_wall_time_secs(results_data)
    state.global_budget = max(0.0, state.global_budget - group_wall_time)
    if _BENCHMARK_MODE == "e1_ml1m" and _E1_RUNTIME is not None:
        state.global_budget = _E1_RUNTIME.budget.remaining_gpu_secs()
    elif _BENCHMARK_MODE == "e2_shared" and _E2_RUNTIME is not None:
        state.global_budget = _E2_RUNTIME.budget.remaining_gpu_secs()

    # --- Auto-generate skill_notes summary ---
    if state.score_history:
        best_ever = max(state.score_history)
        recent_best = state.score_history[-1] if state.score_history else 0.0
        trend = "improving" if len(state.score_history) >= 2 and state.score_history[-1] > state.score_history[-2] else "plateau"
        if textual_gradient_mode:
            state.skill_notes = (
                f"Round {state.round_idx}: best_ever={best_ever:.4f}, "
                f"recent={recent_best:.4f}, trend={trend}. "
                "Textual-gradient routing is active; choose future arms from experiment_skill, "
                "recent_text_gradients, failure_memory, rejected_dims_buffer, and diagnostics."
            )
        else:
            top_arms = sorted(
                [a for a in state.arms if state.arms[a].alpha + state.arms[a].beta > 2],
                key=lambda a: state.arms[a].alpha / (state.arms[a].alpha + state.arms[a].beta),
                reverse=True,
            )[:3]
            top_str = ", ".join(top_arms) if top_arms else "none yet"
            state.skill_notes = (
                f"Round {state.round_idx}: best_ever={best_ever:.4f}, "
                f"recent={recent_best:.4f}, trend={trend}. "
                f"Top arms: {top_str}."
            )

    state.basin_arms[state.current_basin_id] = _clone_arms(state.arms)
    updated_json = state.to_json()

    iteration_log = {
        "iteration": state.round_idx,
        "timestamp": _utc_now_iso(),
        "benchmark_mode": _BENCHMARK_MODE,
        "reward_policy": "validation_only",
        "selection_metric": _SELECTION_METRIC,
        "selection": deepcopy(_LAST_SELECTION_LOG),
        "used_last_raw_results": used_last_raw_results,
        "is_baseline_round": is_baseline_round,
        "promotion_status": {
            "covers_this_group": _promo_covers_this_group,
            "promoted": _promotion_succeeded,
            "trial_group_id": _effective_group_id,
        },
        "budget": {
            "before_secs": budget_before,
            "after_secs": state.global_budget,
            "spent_secs": max(0.0, budget_before - state.global_budget),
            "spend_method": budget_spend_method,
            "group_wall_time_secs": group_wall_time,
        },
        "state_before": state_before_log,
        "state_after": _state_summary(state),
        "trial_results": results_data,
        "rewards": rewards,
        "advantages": advantages,
        "digests": digests,
        "winner": next((d for d in digests if d.get("is_winner")), None),
        "state_update": {
            "alpha_beta_before": arms_before,
            "alpha_beta_after": {
                name: {"alpha": arm.alpha, "beta": arm.beta}
                for name, arm in state.arms.items()
            },
            "updated_arms": arm_updates,
            "score_history_after": list(state.score_history),
            "pending_queue_after": list(state.pending_queue),
        },
        "experiment_skill_update": {
            "new_winner_lesson": new_winner_lesson,
            "new_failure_avoids": new_avoids,
            "recent_text_gradients_after": list(state.recent_text_gradients),
            "failure_memory_after": list(state.failure_memory),
            "experiment_skill_hash_after": _hash_text(state.experiment_skill or ""),
        },
        "errors": _error_counts(results_data),
    }
    _write_json_log(f"iteration_{state.round_idx:04d}.json", iteration_log)
    _write_json_log(f"state/iter_{state.round_idx:04d}_before.json", state_before_log)
    _write_json_log(f"state/iter_{state.round_idx:04d}_after.json", _state_summary(state))
    for idx, result in enumerate(results_data):
        trial_log = {
            "iteration": state.round_idx,
            "trial_id": idx,
            "digest": digests[idx] if idx < len(digests) else None,
            "result": result,
            "reward_used": rewards[idx] if idx < len(rewards) else None,
            "advantage": advantages[idx] if idx < len(advantages) else None,
        }
        _write_json_log(f"trials/iter_{state.round_idx:04d}_trial_{idx}.json", trial_log)

    if _STORE is not None:
        try:
            import datetime
            _STORE.put(_STORE_NAMESPACE, _STORE_KEY, {
                "content": updated_json,
                "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
        except Exception:
            pass

    if _effective_group_id:
        state.processed_trial_group_ids.append(_effective_group_id)
        state.processed_trial_group_ids = state.processed_trial_group_ids[-100:]
        state.basin_arms[state.current_basin_id] = _clone_arms(state.arms)
        updated_json = state.to_json()
        if _STORE is not None:
            try:
                import datetime
                _STORE.put(_STORE_NAMESPACE, _STORE_KEY, {
                    "content": updated_json,
                    "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
            except Exception:
                pass

    return updated_json


# ---------------------------------------------------------------------------
# Tool 4: execute_trial_group  (parallel GRPO group execution)
# ---------------------------------------------------------------------------


def execute_trial_group(
    specs_json: str = "__LAST_PROPOSED__",
    script_path: str = "/workspace/best.py",
    timeout_multiplier: float = 1.5,
    hypotheses_json: str = "",
) -> str:
    """Run one frozen GRPO group using the configured parallel/serial topology."""
    global _LAST_TRIAL_RESULTS, _LAST_TRIAL_GROUP_ID
    script_path = _resolve_script_path(script_path)
    trial_group_id = f"trialgrp_{int(time.time() * 1000)}_{os.getpid()}"
    raw_specs_token = (specs_json or "").strip()
    if (
        _STRICT_VALIDATION_ONLY
        and raw_specs_token not in {"", "__LAST_PROPOSED__", "LAST_PROPOSED"}
    ):
        return _tool_error_results(
            "search-only execution accepts only the cached frozen proposal group; "
            "pass hypotheses separately via hypotheses_json",
            trial_group_id,
            script_path,
        )
    specs, specs_error = _prepare_specs_for_execution(
        specs_json=specs_json,
        hypotheses_json=hypotheses_json,
        trial_group_id=trial_group_id,
        script_path=script_path,
    )
    if specs_error or specs is None:
        return _tool_error_results(specs_error or "invalid specs_json", trial_group_id, script_path)

    jumping_dims = _active_jumping_dims()
    jumping_specs = [
        s for s in specs
        if bool(s.get("is_jumping"))
        or str(s.get("arm", s.get("dimension", ""))) in jumping_dims
        or str(s.get("dimension", "")) in jumping_dims
    ]
    if _STRICT_VALIDATION_ONLY:
        expected_members = [_proposal_member_identity(s) for s in _LAST_PROPOSED_CANDIDATES]
        actual_members = [_proposal_member_identity(s) for s in specs]
        if not expected_members or actual_members != expected_members:
            return _tool_error_results(
                "search-only execution must use the complete frozen proposal group",
                trial_group_id,
                script_path,
            )
        is_baseline_group = all(_is_baseline_spec_data(s) for s in specs)
        if (
            _SERVER_CONFIG.logical_group_size is not None
            and not jumping_specs
            and not is_baseline_group
            and len(specs) != _SERVER_CONFIG.logical_group_size
        ):
            return _tool_error_results(
                "normal search-only execution must preserve logical_group_size",
                trial_group_id,
                script_path,
            )
    if jumping_specs and len(specs) != 1:
        _write_json_log(f"trials/{trial_group_id}_jump_autofix.json", {
            "trial_group_id": trial_group_id,
            "timestamp": _utc_now_iso(),
            "script_path": script_path,
            "reason": "jumping dimensions must run alone; auto-selected the first jumping spec",
            "original_count": len(specs),
            "jumping_count": len(jumping_specs),
            "selected_spec": jumping_specs[0],
        })
        specs = [jumping_specs[0]]
    if jumping_specs:
        specs[0]["is_jumping"] = True

    e1_group_timeout = None
    native_runtime = _E2_RUNTIME if _BENCHMARK_MODE == "e2_shared" else _E1_RUNTIME
    native_mode = _BENCHMARK_MODE in {"e1_ml1m", "e2_shared"}
    if native_mode and not all(_is_baseline_spec_data(s) for s in specs):
        if native_runtime is None:
            raise RuntimeError("Native comparison runtime is not configured")

    workspace_dir = os.path.dirname(script_path)

    try:
        with open(script_path, "rb") as f:
            incumbent_content = f.read()
    except OSError as exc:
        if _BENCHMARK_MODE == "e2_shared":
            # A mistyped tool path is actionable feedback, not a failed model
            # experiment or a reason to discard the cached proposal group.
            return json.dumps({"error": f"Cannot read incumbent: {exc}",
                "retry_script_path": "/workspace/best.py",
                "worker_started": False, "proposal_group_preserved": True})
        raise RuntimeError(f"cannot read incumbent script {script_path}: {exc}") from exc

    try:
        predict_path = os.path.join(workspace_dir, "predict.py")
        with open(predict_path, "r", encoding="utf-8") as f:
            incumbent_predict_content = f.read()
    except OSError:
        incumbent_predict_content = ""

    e1_progress = None
    if e1_group_timeout is None and native_mode and not all(
        _is_baseline_spec_data(s) for s in specs
    ):
        progress_path = native_runtime.budget.path.parent / "pending_group.json"
        group_key = hashlib.sha256(incumbent_content + json.dumps(
            [specs, _LAST_SELECTION_LOG], sort_keys=True).encode()).hexdigest()
        if progress_path.exists():
            saved = json.loads(progress_path.read_text())
            if saved["status"] == "pending":
                if saved["key"] != group_key:
                    raise RuntimeError("Resume the pending native group before proposing another group")
                e1_progress = saved
        if e1_progress is None:
            e1_progress = dict(key=group_key, status="pending", trial_group_id=trial_group_id,
                timeout=native_runtime.group_timeout(len(specs)), members={})
        trial_group_id = e1_progress["trial_group_id"]
        e1_group_timeout = e1_progress["timeout"]

        def save_progress():
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(e1_progress))
            temporary.replace(progress_path)
        save_progress()

    def _run_one(idx: int, spec_data: dict) -> tuple[int, str, dict[str, str]]:
        if e1_progress is not None and str(idx) in e1_progress["members"]:
            saved = e1_progress["members"][str(idx)]
            return idx, saved["result"], saved["payload"]
        slot_dir = os.path.join(workspace_dir, f"_trial_{idx}")
        slot_script = os.path.join(
            slot_dir,
            "best.py" if native_mode else "train.py",
        )
        slot_predict = os.path.join(slot_dir, "predict.py")
        mutated_payload: dict[str, str] = {}
        try:
            if os.path.isdir(slot_dir):
                shutil.rmtree(slot_dir)
            os.makedirs(slot_dir, exist_ok=True)
            for fname in os.listdir(workspace_dir):
                src = os.path.join(workspace_dir, fname)
                dst = os.path.join(slot_dir, fname)
                if fname.startswith("_trial_") or fname == ".git":
                    continue
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            with open(slot_script, "wb") as f:
                f.write(incumbent_content)
            slot_spec_data = deepcopy(spec_data)
            slot_spec_data["trial_group_id"] = trial_group_id
            slot_spec_data["trial_id"] = idx
            if e1_group_timeout is not None:
                slot_spec_data["e1_training_timeout_secs"] = e1_group_timeout
                slot_spec_data["e1_reserved_sibling_starts"] = len(specs) - idx - 1
            runtime_slot_id = (
                0 if _SERVER_CONFIG.trial_group_execution.strip().lower() == "serial" else idx
            )
            result = execute_trial(
                spec_json=json.dumps(slot_spec_data),
                script_path=slot_script,
                timeout_multiplier=timeout_multiplier,
                slot_id=runtime_slot_id,
            )
            try:
                with open(slot_script, "r", encoding="utf-8") as f:
                    mutated_payload["mutated_code_content"] = f.read()
            except OSError:
                mutated_payload["mutated_code_content"] = incumbent_content.decode(errors="replace")
            try:
                with open(slot_predict, "r", encoding="utf-8") as f:
                    mutated_payload["mutated_predict_content"] = f.read()
            except OSError:
                mutated_payload["mutated_predict_content"] = incumbent_predict_content
        finally:
            try:
                shutil.rmtree(slot_dir)
            except OSError:
                pass
        if e1_progress is not None:
            e1_progress["members"][str(idx)] = dict(result=result, payload=mutated_payload)
            save_progress()
        return idx, result, mutated_payload

    group_start = time.time()
    results, mutated_payloads = _execute_trial_members(specs, _run_one)
    group_wall_time_secs = time.time() - group_start

    parsed_results = [json.loads(r) for r in results if r is not None]
    if _STRICT_VALIDATION_ONLY and any(_contains_heldout_feedback(r) for r in parsed_results):
        _LAST_TRIAL_RESULTS = []
        _LAST_TRIAL_GROUP_ID = trial_group_id
        _write_json_log(f"trials/{trial_group_id}_rejected_heldout_feedback.json", {
            "trial_group_id": trial_group_id,
            "timestamp": _utc_now_iso(),
            "reason": "search-only trial execution returned held-out feedback",
            "result_count": len(parsed_results),
        })
        return _tool_error_results(
            "search-only trial execution refused held-out feedback",
            trial_group_id,
            script_path,
        )
    for idx, result in enumerate(parsed_results):
        result["trial_group_id"] = trial_group_id
        result["trial_id"] = idx
        result["group_wall_time_secs"] = group_wall_time_secs
        result["group_size"] = len(specs)
        if idx < len(mutated_payloads) and mutated_payloads[idx]:
            result.update(mutated_payloads[idx] or {})
    _LAST_TRIAL_RESULTS = deepcopy(parsed_results)
    _LAST_TRIAL_GROUP_ID = trial_group_id
    _write_latest_trial_results_log({
        "trial_group_id": trial_group_id,
        "timestamp": _utc_now_iso(),
        "script_path": script_path,
        "selection": deepcopy(_LAST_SELECTION_LOG),
        "group_wall_time_secs": group_wall_time_secs,
        "group_size": len(specs),
        "results": parsed_results,
    })
    if e1_progress is not None:
        e1_progress["status"] = "completed"
        save_progress()
    # Strip large code payloads from the tool output returned to the LLM.
    # _LAST_TRIAL_RESULTS retains the full fields for promote_winner's in-memory path.
    output_results = [
        {k: v for k, v in r.items() if k not in ("mutated_code_content", "mutated_predict_content")}
        for r in parsed_results
    ]
    return json.dumps(output_results, indent=2)


# ---------------------------------------------------------------------------
# Tool 5: promote_winner  (atomic incumbent promotion)
# ---------------------------------------------------------------------------


def promote_winner(
    trial_results_json: str = "",
    script_path: str = "/workspace/best.py",
    winner_idx: int | None = None,
) -> str:
    """Promote the best valid trial to canonical best.py/predict.py/train.py.

    Recommended usage is to omit trial_results_json or pass "__LAST_RESULTS__";
    the tool will read the latest raw execute_trial_group payload from durable
    logs/trials/latest_results.json or the in-memory _LAST_TRIAL_RESULTS cache.
    Raw TrialResult JSON is accepted by the native workflow as a fallback. In
    search-only mode, callers must use the latest cached frozen group. The winner
    is highest valid val_score unless winner_idx is explicitly provided.

    Also sets _LAST_PROMOTION_RESULT so update_thompson_state can gate
    score_history / winner lesson writes on actual promotion success.
    """
    global _LAST_PROMOTION_RESULT
    script_path = _resolve_script_path(script_path)
    _LAST_PROMOTION_RESULT = {}
    raw_results_arg = trial_results_json.strip()
    if (
        _STRICT_VALIDATION_ONLY
        and raw_results_arg not in {"", "__LAST_RESULTS__", "LAST_RESULTS"}
    ):
        fail = {
            "promoted": False,
            "error": "search-only promotion accepts only the latest cached trial group",
        }
        _LAST_PROMOTION_RESULT = fail
        _write_json_log("promotion/rejected_raw_results.json", fail)
        return json.dumps(fail, indent=2)
    requested_group = ""
    if raw_results_arg and raw_results_arg not in {"__LAST_RESULTS__", "LAST_RESULTS"}:
        match = re.search(r'"trial_group_id"\s*:\s*"([^"]+)"', trial_results_json)
        if match:
            requested_group = match.group(1)
    normalised_arg = "" if raw_results_arg in {"__LAST_RESULTS__", "LAST_RESULTS"} else trial_results_json
    results, _used_latest_results = _normalise_trial_results_input(
        normalised_arg,
        expected_group_id=requested_group,
        allow_durable_fallback=not _STRICT_VALIDATION_ONLY,
    )

    if not isinstance(results, list) or not results:
        fail = {"promoted": False, "error": "no trial results available"}
        _LAST_PROMOTION_RESULT = fail
        return json.dumps(fail, indent=2)
    results = _hydrate_trial_results(results)
    if _STRICT_VALIDATION_ONLY:
        strict_error = _strict_group_results_error(results)
        if strict_error:
            fail = {"promoted": False, "error": strict_error}
            _LAST_PROMOTION_RESULT = fail
            _write_json_log("promotion/rejected_search_group.json", fail)
            return json.dumps(fail, indent=2)
    selected_idx = winner_idx if winner_idx is not None else _select_winner_index(results)
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(results):
        fail = {"promoted": False, "error": "no valid winner found"}
        _LAST_PROMOTION_RESULT = fail
        return json.dumps(fail, indent=2)

    winner = results[selected_idx]
    if not _is_valid_trial(winner) or not _has_required_benchmark_score(winner):
        fail = {
            "promoted": False,
            "winner_idx": selected_idx,
            "error": "selected winner is invalid or missing val_score",
        }
        _LAST_PROMOTION_RESULT = fail
        return json.dumps(fail, indent=2)

    spec_block = winner.get("spec") or {}
    winner_arm = spec_block.get("arm", spec_block.get("dimension", ""))
    is_jumping_winner = bool(
        spec_block.get("is_jumping")
        or winner_arm in _active_jumping_dims()
        or spec_block.get("dimension", "") in _active_jumping_dims()
    )
    incumbent_score = _incumbent_score_from_state()
    winner_score = _benchmark_score(winner)
    if not is_jumping_winner and incumbent_score is not None and winner_score <= incumbent_score:
        kept = {
            "promoted": False,
            "kept_incumbent": True,
            "timestamp": _utc_now_iso(),
            "winner_idx": selected_idx,
            "trial_group_id": winner.get("trial_group_id"),
            "val_score": winner.get("val_score"),
            "selection_metric": _SELECTION_METRIC,
            "incumbent_score": incumbent_score,
            "reason": "winner did not exceed incumbent val_score",
            "spec": winner.get("spec"),
        }
        _LAST_PROMOTION_RESULT = kept
        _write_json_log("promotion/latest.json", kept)
        if winner.get("trial_group_id"):
            _write_json_log(f"promotion/{winner.get('trial_group_id')}.json", kept)
        return json.dumps(kept, indent=2)

    workspace_dir = os.path.dirname(script_path)
    predict_path = os.path.join(workspace_dir, "predict.py")
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            current_code = f.read()
    except OSError as exc:
        fail = {"promoted": False, "error": f"cannot read best.py: {exc}"}
        _LAST_PROMOTION_RESULT = fail
        return json.dumps(fail, indent=2)
    try:
        with open(predict_path, "r", encoding="utf-8") as f:
            current_predict = f.read()
    except OSError:
        current_predict = ""

    new_code = winner.get("mutated_code_content")
    new_predict = winner.get("mutated_predict_content")
    if not isinstance(new_code, str) or not new_code.strip():
        new_code, new_predict, err = _apply_spec_to_contents(
            current_code,
            current_predict,
            winner.get("spec") or {},
        )
        if err:
            fail = {
                "promoted": False,
                "winner_idx": selected_idx,
                "error": err,
            }
            _LAST_PROMOTION_RESULT = fail
            return json.dumps(fail, indent=2)
    elif not isinstance(new_predict, str):
        new_predict = current_predict

    backup_code = current_code
    backup_predict = current_predict
    train_path = os.path.join(workspace_dir, "train.py")
    train_py_existed = os.path.exists(train_path)
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        if train_py_existed:
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(new_code)
        if new_predict != current_predict:
            with open(predict_path, "w", encoding="utf-8") as f:
                f.write(new_predict)
    except OSError as exc:
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(backup_code)
            if train_py_existed:
                with open(train_path, "w", encoding="utf-8") as f:
                    f.write(backup_code)
            if backup_predict:
                with open(predict_path, "w", encoding="utf-8") as f:
                    f.write(backup_predict)
        except OSError:
            pass
        fail_log = {"promoted": False, "winner_idx": selected_idx, "error": f"write failed: {exc}"}
        _LAST_PROMOTION_RESULT = fail_log
        return json.dumps(fail_log, indent=2)

    promotion_log = {
        "promoted": True,
        "timestamp": _utc_now_iso(),
        "winner_idx": selected_idx,
        "trial_group_id": winner.get("trial_group_id"),
        "val_score": winner.get("val_score"),
        "val_metrics": winner.get("val_metrics"),
        "selection_metric": _SELECTION_METRIC,
        "spec": winner.get("spec"),
        "provisional_jump": bool(is_jumping_winner and incumbent_score is not None and winner_score <= incumbent_score),
        "incumbent_score": incumbent_score,
        "best_py_updated": new_code != current_code,
        "predict_py_updated": new_predict != current_predict,
        "train_py_updated": train_py_existed and new_code != current_code,
    }
    if promotion_log["provisional_jump"]:
        promotion_log["previous_incumbent_snapshot"] = {
            "score": incumbent_score,
            "script_path": script_path,
            "predict_path": predict_path,
            "train_path": train_path,
            "best_code": backup_code,
            "predict_code": backup_predict,
            "train_code": backup_code if train_py_existed else "",
            "trial_group_id": winner.get("trial_group_id"),
        }
    _LAST_PROMOTION_RESULT = promotion_log
    _write_json_log("promotion/latest.json", promotion_log)
    if winner.get("trial_group_id"):
        _write_json_log(f"promotion/{winner.get('trial_group_id')}.json", promotion_log)
    output_log = deepcopy(promotion_log)
    if "previous_incumbent_snapshot" in output_log:
        snapshot = output_log["previous_incumbent_snapshot"]
        output_log["previous_incumbent_snapshot"] = {
            "score": snapshot.get("score"),
            "trial_group_id": snapshot.get("trial_group_id"),
            "script_path": snapshot.get("script_path"),
            "best_code_chars": len(snapshot.get("best_code", "")),
            "predict_code_chars": len(snapshot.get("predict_code", "")),
            "train_code_chars": len(snapshot.get("train_code", "")),
        }
    return json.dumps(output_log, indent=2)


def _final_runtime_env(slot_id: int = 0) -> dict[str, str]:
    env, _ = _trial_runtime_env(slot_id)
    if _BENCHMARK_MODE == "kuairec" and _GR_TEST_DATA:
        env["GR_TEST_DATA"] = _GR_TEST_DATA
    return env


def _amazon_final_metrics(predict_script: str) -> tuple[float | None, dict | None, dict | None, str | None]:
    try:
        from gagc.benchmarks.amazon_reviews.harness import evaluate
        from gagc.benchmarks.amazon_reviews.task import active_datasets

        primary_metric = os.getenv("GAGC_PRIMARY_METRIC", "HR@10").strip() or "HR@10"
        result = evaluate(
            predict_script=predict_script,
            data_dir=_DATA_DIR,
            test_dir=_TEST_DIR,
            datasets=active_datasets(),
            mode="test",
            max_eval_users=10000,
            verbose=False,
        )
        score = float(result.aggregate.get(primary_metric, result.primary_metric))
        metrics = {ds: r.metrics for ds, r in result.per_dataset.items()}
        return score, metrics, dict(result.aggregate), None
    except Exception as exc:
        return None, None, None, str(exc)


def _amazon_report_metrics(test_metrics: dict | None) -> dict | None:
    if not isinstance(test_metrics, dict):
        return None
    report: dict[str, dict[str, float]] = {}
    for dataset, metrics in test_metrics.items():
        if not isinstance(metrics, dict):
            continue
        report[dataset] = {
            "NDCG@10": _metric_float(metrics.get("NDCG@10"), 0.0),
            "NDCG@20": _metric_float(metrics.get("NDCG@20"), 0.0),
            "HR@10": _metric_float(metrics.get("HR@10", metrics.get("Recall@10")), 0.0),
            "HR@20": _metric_float(metrics.get("HR@20", metrics.get("Recall@20")), 0.0),
        }
    return report


def _final_report_method_label() -> str:
    for key in ("GAGC_REPORT_METHOD", "GAGC_COLD_START", "GAGC_TEMPLATE"):
        value = os.getenv(key, "").strip()
        if value:
            return _pretty_report_method_label(value)
    return "Final"


def _pretty_report_method_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_")
    aliases = {
        "perdataset": "SASRec",
        "sasrec": "SASRec",
        "sasrec_perdataset": "SASRec",
        "sasrec2": "SASRec2",
        "gru4rec": "GRU4Rec",
        "gru4rec_perdataset": "GRU4Rec",
        "bert4rec": "BERT4Rec",
        "bert4rec_perdataset": "BERT4Rec",
        "nextitnet": "NextItNet",
        "nextitnet_perdataset": "NextItNet",
        "hstu": "HSTU",
        "hstu_perdataset": "HSTU",
        "popular": "Popular",
    }
    return aliases.get(normalized, label.strip() or "Final")


def _amazon_report_table_markdown(report_metrics: dict | None, method_label: str | None = None) -> str | None:
    if not isinstance(report_metrics, dict) or not report_metrics:
        return None
    dataset_labels = {
        "Movies_and_TV": "Movies",
        "Industrial_and_Scientific": "Scientific",
        "Electronics": "Electronics",
        "CDs_and_Vinyl": "CDs",
    }
    metric_order = ("NDCG@10", "NDCG@20", "HR@10", "HR@20")
    label = (method_label or _final_report_method_label()).strip() or "Final"
    lines = [f"| Dataset | Metric | {label} |", "|---|---|---:|"]
    for dataset, short_name in dataset_labels.items():
        metrics = report_metrics.get(dataset)
        if not isinstance(metrics, dict):
            continue
        for metric in metric_order:
            lines.append(f"| {short_name} | {metric} | {_metric_float(metrics.get(metric), 0.0):.4f} |")
    return "\n".join(lines) if len(lines) > 2 else None


def _kuairec_final_metrics(stdout: str) -> tuple[float | None, dict | None, dict | None, str | None]:
    score, _, metrics = _run_kuairec_eval(stdout)
    if metrics is None:
        return None, None, None, "could not parse KuaiRec final metrics from stdout"
    aggregate = dict(metrics)
    return score, {"Kuairec": metrics}, aggregate, None


def _kuairec_report_metrics(test_metrics: dict | None) -> dict | None:
    if not isinstance(test_metrics, dict):
        return None
    metrics = test_metrics.get("Kuairec") if isinstance(test_metrics.get("Kuairec"), dict) else test_metrics
    return {
        "Kuairec": {
            "WT-XAUC": _metric_float(metrics.get("WT-XAUC"), 0.0),
            "WT-MAE": _metric_float(metrics.get("WT-MAE"), 0.0),
            "WR-XAUC": _metric_float(metrics.get("WR-XAUC"), 0.0),
            "WR-MAE": _metric_float(metrics.get("WR-MAE"), 0.0),
        }
    }


def evaluate_final_incumbent(
    script_path: str = "/workspace/best.py",
    slot_id: int = 0,
    timeout_secs: float | None = None,
) -> str:
    """Train/evaluate the frozen incumbent once on the held-out test split.

    This tool is for final reporting only. It does not update Thompson state,
    does not promote code, and must not be fed back into future search rounds.
    Amazon returns per-dataset HR/NDCG metrics; KuaiRec returns WT/WR xAUC/MAE.
    """
    script_path = _resolve_script_path(script_path)
    start = time.time()
    workspace_dir = os.path.dirname(script_path)
    predict_path = os.path.join(workspace_dir, "predict.py")
    if not os.path.isfile(script_path):
        result = {"ok": False, "error": f"missing script_path: {script_path}"}
        _write_json_log("final_eval/latest.json", result)
        return json.dumps(result, indent=2)

    env = _final_runtime_env(slot_id)
    timeout = float(timeout_secs) if timeout_secs is not None else _configured_trial_floor_secs() * 2.0
    proc = None
    error: str | None = None
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        error = "timeout"

    stdout = "" if proc is None else proc.stdout.decode(errors="replace")
    stderr = "" if proc is None else proc.stderr.decode(errors="replace")
    if proc is not None and proc.returncode != 0:
        error = stderr[-2000:] if len(stderr) > 2000 else stderr

    test_score: float | None = None
    test_metrics: dict | None = None
    aggregate_metrics: dict | None = None
    metric_error: str | None = None
    if error is None:
        if _BENCHMARK_MODE == "kuairec":
            test_score, test_metrics, aggregate_metrics, metric_error = _kuairec_final_metrics(stdout)
            report_metrics = _kuairec_report_metrics(test_metrics)
            report_table_markdown = None
        else:
            test_score, test_metrics, aggregate_metrics, metric_error = _amazon_final_metrics(predict_path)
            report_metrics = _amazon_report_metrics(test_metrics)
            report_table_markdown = _amazon_report_table_markdown(report_metrics)
        error = metric_error
    else:
        report_metrics = None
        report_table_markdown = None

    result = {
        "ok": error is None,
        "timestamp": _utc_now_iso(),
        "benchmark_mode": _BENCHMARK_MODE,
        "selection_protocol": "final_held_out_test_only",
        "script_path": script_path,
        "test_score": test_score,
        "test_metrics": test_metrics,
        "report_metrics": report_metrics,
        "report_table_markdown": report_table_markdown,
        "aggregate_metrics": aggregate_metrics,
        "wall_time_secs": time.time() - start,
        "error_message": error,
        "stdout_tail": stdout[-3000:] if len(stdout) > 3000 else stdout,
        "stderr_tail": stderr[-2000:] if len(stderr) > 2000 else stderr,
    }
    _write_json_log("final_eval/latest.json", result)
    stamp = str(int(time.time()))
    _write_json_log(f"final_eval/final_{stamp}.json", result)
    return json.dumps(result, indent=2)


# Backward-compatible alias — remove after all callers are updated
update_adamw_state = update_thompson_state
