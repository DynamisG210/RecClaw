from __future__ import annotations

"""RecHarness main orchestrator agent.

Usage (VolcEngine GLM — default):
    from gagc.agent import create_gagc_agent

    agent = create_gagc_agent(
        data_dir        = "./input/trainval",
        cold_start      = "sasrec",   # popular | sasrec | gru4rec | bert4rec
        workspace_root  = "./workspace",
        global_budget_secs = 43200.0,
        num_gpus=8, num_cpus=128,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Optimize the recommender to maximize HR@10."}]},
        config={"configurable": {"thread_id": "run-001"}},
    )

Usage (Claude, override):
    agent = create_gagc_agent(
        llm_provider="anthropic",
        model_id="claude-sonnet-4-6",
        ...
    )
"""

import os
import json
import shutil
import sqlite3
import sys
from typing import Any

from gagc.prompts import (
    E1_ORCHESTRATOR_PROMPT,
    E2_ORCHESTRATOR_PROMPT,
    GAGC_ORCHESTRATOR_PROMPT,
    GR_ORCHESTRATOR_PROMPT,
)
from gagc.templates import get_template_dir
from gagc.tools import (
    ServerConfig,
    default_experiment_skill_for_policy,
    execute_trial,
    execute_trial_group,
    evaluate_final_incumbent,
    promote_winner,
    propose_action_group,
    update_thompson_state,
    _active_arms,
)
import gagc.tools as _tools_module
# Anthropic-compatible: https://ark.cn-beijing.volces.com/api/plan
# OpenAI-compatible:    https://ark.cn-beijing.volces.com/api/plan/v3
# Using OpenAI-compatible since langchain_openai is more reliable for tool-use.
_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
_VOLCENGINE_MODEL    = "glm-5.2[1m]"


def _gagc_agent_tools(enable_final_evaluation: bool) -> list[Any]:
    """Return the native tool surface, optionally withholding held-out access."""
    agent_tools = [
        propose_action_group,
        execute_trial_group,
        promote_winner,
        update_thompson_state,
    ]
    if enable_final_evaluation:
        agent_tools.insert(2, execute_trial)
        agent_tools.append(evaluate_final_incumbent)
    return agent_tools


def _gagc_system_prompt(
    enable_final_evaluation: bool,
    benchmark_mode: str = "amazon_reviews",
) -> str:
    """Remove final-test instructions when constructing a search-only agent."""
    if benchmark_mode == "e2_shared":
        if enable_final_evaluation:
            raise ValueError("E2 search agents cannot expose final evaluation")
        return E2_ORCHESTRATOR_PROMPT
    if benchmark_mode == "e1_ml1m":
        if enable_final_evaluation:
            raise ValueError("E1 search agents cannot expose final evaluation")
        return E1_ORCHESTRATOR_PROMPT
    if enable_final_evaluation:
        return GAGC_ORCHESTRATOR_PROMPT

    search_prompt = GAGC_ORCHESTRATOR_PROMPT.replace(
        " / `evaluate_final_incumbent` script_path",
        " script_path",
    )
    search_only_replacements = {
        "and 16 NUMA-local CPU cores set automatically by execute_trial.":
            "and 16 NUMA-local CPU cores set automatically by the internal trial executor.",
        "During search, execute_trial automatically calls only:":
            "During search, each execute_trial_group member automatically calls only:",
        "- `execute_trial` / `execute_trial_group` / `promote_winner` script_path":
            "- `execute_trial_group` / `promote_winner` script_path",
        "execute_trial returns a JSON object with:":
            "Each execute_trial_group member returns a JSON object with:",
        "4. **Run trials** -- for mutation rounds, call `execute_trial_group` using the cached candidates":
            "4. **Run trials** -- for baseline and mutation rounds, call `execute_trial_group` using the cached candidates",
        (
            "   Never call execute_trial for a mutation; execute_trial is reserved for the no-op baseline\n"
            "   on canonical `/workspace/best.py`."
        ): (
            "   The mandatory no-op baseline also uses this cached-group call with its singleton\n"
            "   proposal; it remains one baseline round and does not update Thompson alpha/beta."
        ),
        "from the tool cache / `trials/latest_results.json`.":
            "from the private in-process tool cache.",
    }
    for native_text, search_only_text in search_only_replacements.items():
        if native_text not in search_prompt:
            raise RuntimeError(
                f"RecHarness search-only prompt contract changed upstream: {native_text!r}"
            )
        search_prompt = search_prompt.replace(native_text, search_only_text)
    final_instruction = (
        "5. After the search budget is exhausted or the user asks for final reporting, call\n"
        "   `evaluate_final_incumbent(script_path)` exactly once on the frozen incumbent and\n"
        "   report the returned test metrics. For Amazon, paste `report_table_markdown` as the\n"
        "   final test-results table instead of hand-writing a wide table. Do not feed final\n"
        "   test results into another round."
    )
    search_only_instruction = (
        "5. After the search budget is exhausted, stop with the frozen incumbent identity and\n"
        "   validation-only audit record. Final evaluation belongs to a separate trusted process;\n"
        "   do not request, infer, or report held-out metrics from this agent."
    )
    if final_instruction not in search_prompt:
        raise RuntimeError("RecHarness final-evaluation prompt contract changed upstream")
    return search_prompt.replace(final_instruction, search_only_instruction)


def _resolve_volcengine_api_key(api_key: str | None) -> str:
    """Resolve the VolcEngine API key from the argument or environment.

    The key is never hardcoded in source. Provide it via the ``api_key``
    argument or the ``VOLCENGINE_API_KEY`` / ``ARK_API_KEY`` environment
    variable (see ``.env.example``).
    """
    key = api_key or os.environ.get("VOLCENGINE_API_KEY") or os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError(
            "VolcEngine API key not configured. Set VOLCENGINE_API_KEY (see "
            ".env.example) or pass api_key=... to create_gagc_agent."
        )
    return key


TEXTUAL_GRADIENT_ROUTING_PROMPT = """

# Textual-gradient routing ablation

Runtime `routing_policy` is `textual_gradient`, so Thompson Sampling is disabled for arm routing.
Before calling `propose_action_group`, choose the next arms yourself from `/thompson_state/state.json`
using `skill_notes`, `experiment_skill`, `recent_text_gradients`, `failure_memory`,
`rejected_dims_buffer`, score history, and diagnostics. Pass them as `textual_selected_arms_json`,
for example:
`{ "arms": [{"arm": "tune_lr", "delta": -1, "reason": "recent loss oscillation suggests lower LR"}, {"arm": "tune_dropout", "delta": 1}] }`.
Do not use α/β values to rank arms in this mode; they are retained only for logging compatibility.
"""


def _build_model(llm_provider: str, model_id: str, api_key: str | None, base_url: str | None):
    """Construct the LangChain chat model for the given provider."""
    max_tokens = int(os.environ.get("GAGC_LLM_MAX_TOKENS", "4096"))
    if llm_provider == "volcengine":
        # VolcEngine Ark is OpenAI-compatible — use ChatOpenAI, not ChatAnthropic.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=_resolve_volcengine_api_key(api_key),
            base_url=base_url or os.environ.get("VOLCENGINE_BASE_URL") or _VOLCENGINE_BASE_URL,
            max_tokens=max_tokens,
        )
    elif llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_id,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            max_tokens=max_tokens,
        )
    else:
        # Generic: pass a "provider:model" string directly to deepagents
        return f"{llm_provider}:{model_id}"


def _enable_claude_backend_by_default() -> None:
    raw = os.environ.get("GAGC_ENABLE_CLAUDE_BACKEND")
    if raw is None:
        os.environ["GAGC_ENABLE_CLAUDE_BACKEND"] = "1"
        return
    if raw.strip().lower() in {"0", "false", "no", "off", "none"}:
        return
    os.environ["GAGC_ENABLE_CLAUDE_BACKEND"] = raw


def _init_workspace(workspace_root: str, cold_start: str) -> None:
    """Copy cold-start template files into workspace_root if best.py is absent."""
    best_py = os.path.join(workspace_root, "best.py")
    if os.path.exists(best_py):
        return  # already initialised — don't overwrite

    template_dir = get_template_dir(cold_start)
    os.makedirs(workspace_root, exist_ok=True)
    for fname in os.listdir(template_dir):
        src = os.path.join(template_dir, fname)
        dst = os.path.join(workspace_root, fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # The entry point RecHarness evolves is always best.py
    train_src = os.path.join(workspace_root, "train.py")
    if os.path.exists(train_src) and not os.path.exists(best_py):
        shutil.copy2(train_src, best_py)

    # Initialize git repo so agent can use git diff to generate patches
    import subprocess
    try:
        subprocess.run(
            ["git", "init"],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial cold-start template"],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Git init failed — agent will have to generate diffs another way
        pass


class _PersistentStoreProxy:
    """Mirror the Thompson state store to disk without changing StoreBackend use."""

    def __init__(self, inner: Any, path: str):
        self._inner = inner
        self._path = os.path.abspath(path)
        self._load()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                value = json.load(f)
            if not isinstance(value, dict):
                return
            self._inner.put(("gagc", "thompson"), "/state.json", value)
        except Exception as exc:  # pragma: no cover - best-effort resume aid
            print(f"[gagc] warning: failed to load Thompson state {self._path}: {exc}", file=sys.stderr)

    def _persist(self, namespace: Any, key: Any, value: Any) -> None:
        if tuple(namespace) != ("gagc", "thompson") or key != "/state.json":
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp_path = f"{self._path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # pragma: no cover - best-effort resume aid
            print(f"[gagc] warning: failed to persist Thompson state {self._path}: {exc}", file=sys.stderr)

    def put(self, namespace: Any, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._inner.put(namespace, key, value, *args, **kwargs)
        self._persist(namespace, key, value)
        return result

    def get(self, namespace: Any, key: str, *args: Any, **kwargs: Any) -> Any:
        return self._inner.get(namespace, key, *args, **kwargs)


def _checkpoint_path(logs_root: str, checkpoint_path: str | None) -> str | None:
    raw = checkpoint_path or os.environ.get("GAGC_LANGGRAPH_CHECKPOINT_PATH") or os.environ.get("GAGC_CHECKPOINT_PATH")
    if raw and raw.strip().lower() in {"", "0", "false", "none", "memory", "off"}:
        return None
    return os.path.abspath(raw or os.path.join(logs_root, "langgraph_checkpoints.sqlite"))


def _build_checkpointer(logs_root: str, checkpoint_path: str | None, enable_persistent_checkpoint: bool):
    from langgraph.checkpoint.memory import MemorySaver

    if not enable_persistent_checkpoint:
        return MemorySaver()

    path = _checkpoint_path(logs_root, checkpoint_path)
    if not path:
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as exc:
        print(
            "[gagc] warning: langgraph-checkpoint-sqlite is unavailable; "
            f"falling back to in-memory checkpoints ({exc})",
            file=sys.stderr,
        )
        return MemorySaver()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    if hasattr(saver, "setup"):
        saver.setup()
    return saver


def _build_store(logs_root: str, thompson_state_path: str | None, enable_persistent_checkpoint: bool):
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    if not enable_persistent_checkpoint:
        return store
    path = thompson_state_path or os.environ.get("GAGC_THOMPSON_STATE_PATH")
    if path and path.strip().lower() in {"", "0", "false", "none", "memory", "off"}:
        return store
    return _PersistentStoreProxy(store, path or os.path.join(logs_root, "thompson_state.json"))


def _seed_thompson_state(store: Any, default_state: Any) -> None:
    try:
        item = store.get(("gagc", "thompson"), "/state.json")
        if item and item.value.get("content"):
            return
    except Exception:
        pass
    import datetime
    store.put(("gagc", "thompson"), "/state.json", {
        "content": default_state.to_json(),
        "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


def create_gagc_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "volcengine",
    model_id: str = _VOLCENGINE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    data_dir: str = "./input/trainval",
    test_dir: str | None = None,
    cold_start: str = "sasrec",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 8,
    num_cpus: int = 128,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
    # ── E1-safe execution extensions (appended for positional compatibility) ──
    enable_final_evaluation: bool = True,
    trial_group_execution: str = "parallel",
    logical_group_size: int | None = None,
    benchmark_mode: str = "amazon_reviews",
    selection_metric: str = "val_score",
    e1_data_root: str = "",
    e1_manifest: str = "",
    e1_manifest_sha256: str = "",
    e1_device: str = "auto",
    e1_runtime: Any = None,
    e1_model: Any = None,
    e2_runtime: Any = None,
    e2_model: Any = None,
):
    """Build and return the RecHarness orchestrator as a compiled deep agent.

    Args:
        llm_provider: 'volcengine' (default, GLM-5.1) | 'anthropic' | any deepagents provider string.
        model_id: Model name. Default 'glm-5.2' for VolcEngine.
        api_key: API key. For VolcEngine, read from VOLCENGINE_API_KEY env var if not passed.
        base_url: Override base URL. VolcEngine URL is bundled as default.
        data_dir: Directory with *_train.txt and *_valid.txt files.
        test_dir: Directory with *_test.txt files (defaults to data_dir).
        enable_final_evaluation: When False, reject a test_dir, omit held-out
            paths from the prompt and trial environment, and do not expose the
            final-evaluation tool. The execution sandbox must also omit any test
            mount; this flag alone is not a filesystem security boundary.
        cold_start: Template name — 'popular' | 'sasrec' | 'gru4rec' | 'bert4rec'.
        workspace_root: Root for training scripts.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total GPU/compute-time budget in seconds (default 43,200 s).
        num_gpus: GPU devices available (default 8, A800 server).
        num_cpus: CPU cores available (default 128, A800 server).
        trial_group_execution: ``parallel`` for the native topology or
            ``serial`` to run every member of one frozen logical group on one
            accelerator before a single promotion/posterior update.
        logical_group_size: Optional enforced size for normal Thompson groups.
            Use 4 for one-GPU E1 execution; jump rounds remain native singletons.
        benchmark_mode: Internal benchmark adapter name. Existing callers keep
            the native ``amazon_reviews`` mode; use ``create_e1_agent`` for E1.
        selection_metric: Validation metric label recorded in search logs.
        e1_data_root: E1 train/validation data root passed to the fixed runner.
        e1_manifest: E1 split-manifest path passed to the fixed runner.
        e1_manifest_sha256: Frozen E1 manifest digest passed to the fixed runner.
        e1_device: E1 runner device policy: auto, cpu, or cuda.
        use_thompson_sampling: If False, run the textual-gradient ablation:
            the LLM chooses arms from global experiment memory and Thompson
            posterior samples are not used for routing.
        basin_transfer_rho: Optional one-time posterior transfer coefficient
            used when a successful basin jump creates a new local Thompson basin.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.
    """
    benchmark_mode = benchmark_mode.strip().lower()
    if benchmark_mode not in {"amazon_reviews", "e1_ml1m", "e2_shared"}:
        raise ValueError(f"unsupported benchmark_mode: {benchmark_mode!r}")
    native_mode = benchmark_mode in {"e1_ml1m", "e2_shared"}
    native_runtime = e2_runtime if benchmark_mode == "e2_shared" else e1_runtime
    native_model = e2_model if benchmark_mode == "e2_shared" else e1_model
    execution_mode = trial_group_execution.strip().lower()
    if execution_mode not in {"parallel", "serial"}:
        raise ValueError("trial_group_execution must be 'parallel' or 'serial'")
    if execution_mode == "serial" and num_gpus != 1:
        raise ValueError("serial trial-group execution requires num_gpus=1")
    if execution_mode == "serial" and (gpu_ids is None or len(gpu_ids) != 1):
        raise ValueError(
            "serial trial-group execution requires exactly one explicit gpu_id"
        )
    if execution_mode == "serial" and (
        isinstance(gpu_ids[0], bool)
        or not isinstance(gpu_ids[0], int)
        or gpu_ids[0] < 0
    ):
        raise ValueError(
            "serial trial-group execution gpu_id must be one nonnegative integer"
        )
    if logical_group_size is not None and logical_group_size < 2:
        raise ValueError("logical_group_size must be at least 2")
    if not enable_final_evaluation and test_dir is not None:
        raise ValueError(
            "test_dir must be None when enable_final_evaluation is False; "
            "the search runtime must not receive a held-out path"
        )
    if benchmark_mode == "e1_ml1m":
        if enable_final_evaluation:
            raise ValueError("E1 is search-only; final evaluation belongs to a separate process")
        if execution_mode != "serial" or num_gpus != 1 or logical_group_size != 4:
            raise ValueError("E1 requires one-GPU serial execution with logical_group_size=4")
        if cold_start != "e1_multvae":
            raise ValueError("E1 requires the frozen e1_multvae cold-start template")
        if selection_metric != "NDCG@10":
            raise ValueError("E1 selection_metric must be NDCG@10")
        if not e1_data_root.strip():
            raise ValueError("E1_DATA_ROOT must be configured")
        if not e1_manifest.strip():
            raise ValueError("E1_MANIFEST must be configured")
        normalised_manifest_sha256 = e1_manifest_sha256.strip().lower()
        if (
            len(normalised_manifest_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in normalised_manifest_sha256)
        ):
            raise ValueError("E1_MANIFEST_SHA256 must be a 64-character hexadecimal digest")
        if e1_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("e1_device must be 'auto', 'cpu', or 'cuda'")
    if benchmark_mode == "e2_shared":
        if enable_final_evaluation or selection_metric != "NDCG@5" or cold_start != "e2_shared_sasrec":
            raise ValueError("E2 requires its shared SASRec parent, validation NDCG@5 and no final evaluator")
        if execution_mode != "serial" or num_gpus != 1 or logical_group_size != 4:
            raise ValueError("E2 requires one-GPU serial execution with native groups of four")
    if not native_mode:
        _enable_claude_backend_by_default()
    elif native_runtime is None or native_model is None:
        raise ValueError("Native comparison requires its configured endpoint2 model and task runtime")

    # --- hardware config ---
    _tools_module._BENCHMARK_MODE = benchmark_mode
    _tools_module._GR_TRAIN_DATA = ""
    _tools_module._GR_VAL_DATA = ""
    _tools_module._GR_TEST_DATA = ""
    _tools_module._E1_DATA_ROOT = os.path.abspath(e1_data_root) if e1_data_root else ""
    _tools_module._E1_SEARCH_ROOT = os.path.abspath(workspace_root) if benchmark_mode == "e1_ml1m" else ""
    _tools_module._E1_MANIFEST = os.path.abspath(e1_manifest) if e1_manifest else ""
    _tools_module._E1_MANIFEST_SHA256 = e1_manifest_sha256.strip().lower()
    _tools_module._E1_DEVICE = e1_device
    _tools_module._E1_RUNTIME = e1_runtime
    _tools_module._E2_RUNTIME = e2_runtime
    _tools_module._SELECTION_METRIC = selection_metric
    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // num_gpus,
        gpu_ids=gpu_ids or [],
        trial_group_execution=execution_mode,
        logical_group_size=logical_group_size,
    )
    _tools_module._DATA_DIR = "" if native_mode else data_dir
    _tools_module._TEST_DIR = (test_dir or data_dir) if enable_final_evaluation else ""
    _tools_module._STRICT_VALIDATION_ONLY = not enable_final_evaluation
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start
    if basin_transfer_rho is not None:
        _tools_module.BASIN_TRANSFER_RHO = min(1.0, max(0.0, float(basin_transfer_rho)))

    os.makedirs(logs_root, exist_ok=True)

    # --- cold start ---
    if native_mode:
        from pathlib import Path
        os.makedirs(workspace_root, exist_ok=True)
        best = Path(workspace_root) / "best.py"
        if not best.exists():
            best.write_text(native_runtime.parent_source, encoding="utf-8")
    else:
        _init_workspace(workspace_root, cold_start)

    # absolute paths
    abs_workspace = os.path.abspath(workspace_root)
    abs_data_dir  = os.path.abspath(data_dir)
    abs_test_dir = os.path.abspath(test_dir or data_dir) if enable_final_evaluation else None
    abs_logs_root = os.path.abspath(logs_root)

    if enable_final_evaluation:
        final_eval_context = f"- test_dir       : {abs_test_dir}\n"
    else:
        final_eval_context = (
            "- final test     : unavailable in this search-only agent; validation feedback only\n"
        )
    if benchmark_mode == "e2_shared":
        workspace_files_context = (
            "- /workspace/best.py — opaque JSON of the complete four-file E2 program; never execute\n"
        )
        data_context = "- selection: shared CDs/SASRec frozen sampled validation NDCG@5\n"
    elif benchmark_mode == "e1_ml1m":
        workspace_files_context = (
            "- /workspace/best.py          — E1 candidate module (the only evolvable file)\n"
        )
        data_context = (
            f"- E1_DATA_ROOT  : {_tools_module._E1_DATA_ROOT}\n"
            f"- E1_MANIFEST   : {_tools_module._E1_MANIFEST}\n"
            f"- E1_SEARCH_ROOT: {abs_workspace}\n"
            f"- selection     : NDCG@10 on validation only\n"
        )
    else:
        workspace_files_context = (
            "- /workspace/best.py          — training script\n"
            "- /workspace/predict.py       — inference script\n"
        )
        data_context = f"- data_dir       : {abs_data_dir}\n"

    diff_context = (
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff best.py\") to generate code_diff)\n"
        if not native_mode else ""
    )
    system_prompt = (
        _gagc_system_prompt(enable_final_evaluation, benchmark_mode)
        + f"\n\n# Runtime context\n"
        f"## Virtual paths (use these with read_file / write_file / ls)\n"
        + workspace_files_context
        + f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths (use for trial-tool script_path)\n"
        f"- trial-tool script_path      : {abs_workspace}/best.py\n"
        + diff_context + f"\n## Other\n"
        + data_context
        + final_eval_context
        + f"- global_budget  : {global_budget_secs}s\n"
        f"- num_gpus       : {num_gpus}\n"
        f"- gpu_ids        : {gpu_ids or list(range(num_gpus))}\n"
        f"- cold_start     : {cold_start}\n"
        f"\nRULE: read_file/write_file/ls MUST use the virtual paths above (starting with\n"
        f"/workspace/, /thompson_state/, /logs/). NEVER pass a real absolute path like\n"
        f"{abs_workspace}/... to read_file — it will not be found.\n"
        + ("Source/state views are read-only; mutations and promotion belong to trusted tools.\n"
           if native_mode else
           "The ONLY exceptions are trial-tool script_path and bash commands for git diff.\n")
    )
    if benchmark_mode == "e1_ml1m":
        system_prompt += e1_runtime.pilot_context
    elif benchmark_mode == "e2_shared":
        from gagc.e2_task import SCOPE_NOTE
        system_prompt += "\n\n" + SCOPE_NOTE + "\n"
        system_prompt += e2_runtime.shared_context
    if (
        benchmark_mode == "amazon_reviews"
        and os.environ.get("GAGC_FULL_RANKING", "").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        system_prompt += (
            "\n## Runtime evaluation override\n"
            "This run uses full-item ranking over all non-history items. "
            "Treat Recall@10 as the primary metric, and report Recall@5/10 and NDCG@5/10. "
            "In this leave-one-out setting, internal HR@K hits are exposed as Recall@K aliases.\n"
        )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    # Patch deepagents summarization to prevent empty message list reaching the LLM.
    #
    # Root cause: after summarization, _apply_event_to_messages may return only
    # [summary_msg] when cutoff_idx==len(messages). Then if _truncate_args further
    # reduces messages, or _find_safe_cutoff_point returns idx==len(messages) (when
    # the entire tail is ToolMessages with no matching AIMessage), the effective
    # message list passed to handler can be [] — the LLM API returns empty generations
    # → .generations[0][0] raises IndexError.
    #
    # Fix: patch _get_effective_messages in _DeepAgentsSummarizationMiddleware to
    # guarantee the result is never empty (fall back to the raw request messages).
    from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware as _DSM
    _original_get_effective = _DSM._get_effective_messages
    def _safe_get_effective_messages(self, request):  # type: ignore[misc]
        result = _original_get_effective(self, request)
        if not result:
            # Fallback: return raw messages; at minimum one message must exist.
            return list(request.messages) or [request.system_message] if request.system_message else list(request.messages)
        return result
    _DSM._get_effective_messages = _safe_get_effective_messages  # type: ignore[assignment]

    # Also patch _find_safe_cutoff_point so it never returns len(messages),
    # which would cause _partition_messages to produce an empty preserved slice.
    from langchain.agents.middleware.summarization import SummarizationMiddleware as _SM
    _original_cutoff = _SM._find_safe_cutoff_point
    @staticmethod  # type: ignore[misc]
    def _safe_cutoff_point(messages, cutoff_index):
        result = _original_cutoff(messages, cutoff_index)
        return min(result, max(0, len(messages) - 1))
    _SM._find_safe_cutoff_point = _safe_cutoff_point  # type: ignore[assignment]

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = native_model if native_mode else _build_model(llm_provider, model_id, api_key, base_url)

    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)

    # Inject store into tools module so update_thompson_state can persist state
    # directly without requiring the LLM to call write_file.
    _tools_module._STORE = store

    # Pre-seed ThompsonState so read_file("/thompson_state/state.json") succeeds on iteration 1.
    from gagc.state import ThompsonState, ArmState
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        recent_text_gradients=[],
        failure_memory=[],
        experiment_memory=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    (FilesystemBackend(root_dir=abs_workspace, virtual_mode=True)
                               if native_mode else
                               LocalShellBackend(root_dir=abs_workspace, virtual_mode=True)),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    if native_mode:
        from gagc.e1_runtime import readonly_backend
        backend = readonly_backend({
            "/workspace/": FilesystemBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/": StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/": FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        })

    middleware = []
    if benchmark_mode == "e2_shared":
        from gagc.e2_context import context_middleware
        middleware = [context_middleware(model, abs_logs_root)]

    agent = create_deep_agent(
        model=model,
        tools=_gagc_agent_tools(enable_final_evaluation),
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
        middleware=middleware,
    )

    return agent


def create_e1_agent(
    *,
    base_url: str,
    api_key: str,
    model: str,
    native_runner: Any,
    budget: Any = None,
    baseline: dict,
    parent_source: str,
    pilot_context: str = "",
    critic_model: str = "gpt-5.6-luna",
    data_root: str | None = None,
    manifest: str | None = None,
    manifest_sha256: str | None = None,
    workspace_root: str | None = None,
    logs_root: str = "./logs_e1",
    global_budget_secs: float = 14400.0,
    num_cpus: int = 16,
    gpu_id: int = 0,
    device: str = "auto",
    test_dir: str | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build the search-only RecHarness agent for the E1 MovieLens-1M task.

    ``base_url``, ``api_key``, and ``model`` are deliberately required keyword
    arguments for the endpoint2 OpenAI-compatible route. The key is passed only
    to the in-memory chat-model constructor; this factory neither exports it to
    the trial environment nor writes it to workspace, logs, or checkpoints.

    Dataset and manifest values may be supplied explicitly or through
    ``E1_DATA_ROOT``, ``E1_MANIFEST``, and ``E1_MANIFEST_SHA256``. The search
    root is the canonical workspace (or ``E1_SEARCH_ROOT`` when no explicit
    workspace is supplied). E1 always uses one physical GPU, serial execution,
    and normal logical groups of four candidates (native jumps stay singletons).
    ``native_runner`` must come from the shared RO ``make_native_runner``;
    ``baseline`` is its previously measured shared parent result. No baseline
    training or API request occurs when this factory constructs the graph.
    """
    if test_dir is not None:
        raise ValueError(
            "test_dir is forbidden for E1 search; final evaluation runs in a separate process"
        )
    for field_name, value in (
        ("base_url", base_url),
        ("api_key", api_key),
        ("model", model),
    ):
        if not value or not value.strip():
            raise ValueError(f"endpoint2 {field_name} must be explicit")

    resolved_data_root = data_root or os.environ.get("E1_DATA_ROOT", "")
    resolved_manifest = manifest or os.environ.get("E1_MANIFEST", "")
    resolved_manifest_sha256 = (
        manifest_sha256 or os.environ.get("E1_MANIFEST_SHA256", "")
    ).strip().lower()
    resolved_workspace = (
        workspace_root or os.environ.get("E1_SEARCH_ROOT") or "./workspace_e1"
    )
    if not resolved_data_root:
        raise ValueError("E1_DATA_ROOT must be supplied or set in the environment")
    if not resolved_manifest:
        raise ValueError("E1_MANIFEST must be supplied or set in the environment")
    if (
        len(resolved_manifest_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in resolved_manifest_sha256)
    ):
        raise ValueError("E1_MANIFEST_SHA256 must be a 64-character hexadecimal digest")

    resolved_device = device.strip().lower()
    if resolved_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    from pathlib import Path
    import random
    from gagc.e1_runtime import Budget, E1Runtime, endpoint2_model
    if global_budget_secs != 14400.0:
        raise ValueError("E1 GPU cap is frozen at 14400 seconds")
    if Path(logs_root).resolve().is_relative_to(Path(resolved_workspace).resolve()):
        raise ValueError("E1 trusted logs/ledger must be outside the candidate workspace")
    random.seed(54202)
    budget = budget if budget is not None else Budget(Path(logs_root) / "budget.json")
    controller_model = endpoint2_model(base_url=base_url, api_key=api_key, model=model, budget=budget)
    critic = endpoint2_model(base_url=base_url, api_key=api_key, model=critic_model, budget=budget)
    runtime = E1Runtime(budget=budget, implementer=controller_model, critic=critic,
                       native_runner=native_runner, baseline=baseline, parent_source=parent_source)
    runtime.pilot_context = pilot_context

    return create_gagc_agent(
        llm_provider="openai_compatible",
        model_id=model.strip(),
        api_key=api_key,
        base_url=base_url.strip(),
        data_dir=os.path.abspath(resolved_data_root),
        test_dir=None,
        cold_start="e1_multvae",
        workspace_root=os.path.abspath(resolved_workspace),
        logs_root=logs_root,
        global_budget_secs=global_budget_secs,
        num_gpus=1,
        num_cpus=num_cpus,
        gpu_ids=[gpu_id],
        use_thompson_sampling=use_thompson_sampling,
        basin_transfer_rho=basin_transfer_rho,
        enable_persistent_checkpoint=enable_persistent_checkpoint,
        checkpoint_path=checkpoint_path,
        thompson_state_path=thompson_state_path,
        enable_final_evaluation=False,
        trial_group_execution="serial",
        logical_group_size=4,
        benchmark_mode="e1_ml1m",
        selection_metric="NDCG@10",
        e1_data_root=os.path.abspath(resolved_data_root),
        e1_manifest=os.path.abspath(resolved_manifest),
        e1_manifest_sha256=resolved_manifest_sha256,
        e1_device=resolved_device,
        e1_runtime=runtime,
        e1_model=controller_model,
    )


def create_gr_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "volcengine",
    model_id: str = _VOLCENGINE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    train_data: str = "./input/train_data.npy",
    val_data: str | None = None,
    test_data: str = "./input/test_data.npy",
    cold_start: str = "gr",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 8,
    num_cpus: int = 128,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build and return the RecHarness GR orchestrator for KuaiRec watch-time prediction.

    Args:
        llm_provider: 'volcengine' (default) | 'anthropic'.
        model_id: Model name.
        train_data: Path to train .npy file.
        val_data: Path to validation .npy file used during search. Defaults to
            GAGC_GR_VAL_DATA, then test_data for backward compatibility.
        test_data: Path to held-out test .npy file used only by final evaluation.
        cold_start: KuaiRec template name: 'gr' | 'd2q' | 'ks_d2q' | 'tpm'.
        workspace_root: Root for training scripts.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total GPU/compute-time budget.
        num_gpus: GPU devices available.
        num_cpus: CPU cores available.
        gpu_ids: Explicit GPU IDs to use (defaults to range(num_gpus)).
        use_thompson_sampling: If False, run the textual-gradient ablation:
            the LLM chooses arms from global experiment memory and Thompson
            posterior samples are not used for routing.
        basin_transfer_rho: Optional one-time posterior transfer coefficient
            used when a successful basin jump creates a new local Thompson basin.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.

    Usage:
        from gagc.agent import create_gr_agent
        agent = create_gr_agent(
            train_data="./input/train_data.npy",
            test_data="./input/test_data.npy",
            workspace_root="./workspace_gr",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Optimize the GR model to maximize xAUC and minimize MAE."}]},
            config={"configurable": {"thread_id": "gr-run-001"}},
        )
    """
    # Switch benchmark mode to kuairec BEFORE init_workspace / pre-seed
    _tools_module._BENCHMARK_MODE = "kuairec"
    _tools_module._DATA_DIR = ""
    _tools_module._TEST_DIR = ""
    _tools_module._STRICT_VALIDATION_ONLY = False
    _tools_module._SELECTION_METRIC = "val_score"
    _tools_module._E1_DATA_ROOT = ""
    _tools_module._E1_SEARCH_ROOT = ""
    _tools_module._E1_MANIFEST = ""
    _tools_module._E1_MANIFEST_SHA256 = ""
    _tools_module._E1_DEVICE = "auto"
    _tools_module._GR_TRAIN_DATA = os.path.abspath(train_data)
    _tools_module._GR_VAL_DATA = os.path.abspath(val_data or os.environ.get("GAGC_GR_VAL_DATA") or test_data)
    _tools_module._GR_TEST_DATA = os.path.abspath(test_data)
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start
    if basin_transfer_rho is not None:
        _tools_module.BASIN_TRANSFER_RHO = min(1.0, max(0.0, float(basin_transfer_rho)))

    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // max(num_gpus, 1),
        gpu_ids=gpu_ids or [],
    )
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)

    os.makedirs(logs_root, exist_ok=True)
    _init_workspace(workspace_root, cold_start)

    abs_workspace = os.path.abspath(workspace_root)
    abs_logs_root = os.path.abspath(logs_root)

    system_prompt = (
        GR_ORCHESTRATOR_PROMPT
        + f"\n\n# Runtime context\n"
        f"## Virtual paths\n"
        f"- /workspace/best.py          — KuaiRec training script (RecHarness evolves this)\n"
        f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths\n"
        f"- trial-tool script_path      : {abs_workspace}/best.py\n"
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff best.py\") to generate code_diff)\n"
        f"\n## Data\n"
        f"- GR_TRAIN_DATA : {_tools_module._GR_TRAIN_DATA}\n"
        f"- GR_VAL_DATA   : {_tools_module._GR_VAL_DATA} (used during search)\n"
        f"- GR_TEST_DATA  : {_tools_module._GR_TEST_DATA} (held out until evaluate_final_incumbent)\n"
        f"- cold_start    : {cold_start}\n"
        f"\n## Budget\n"
        f"- global_budget : {global_budget_secs}s\n"
        f"- num_gpus      : {num_gpus}\n"
        f"- gpu_ids       : {gpu_ids or list(range(num_gpus))}\n"
        f"\nRULE: read_file/write_file/ls MUST use virtual paths (/workspace/, /thompson_state/, /logs/).\n"
        f"NEVER pass a real absolute path to read_file. Use the real trial-tool script_path above for execute_trial, execute_trial_group, promote_winner, and evaluate_final_incumbent.\n"
    )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware as _DSM
    _original_get_effective = _DSM._get_effective_messages
    def _safe_get_effective_messages(self, request):  # type: ignore[misc]
        result = _original_get_effective(self, request)
        if not result:
            return list(request.messages) or [request.system_message] if request.system_message else list(request.messages)
        return result
    _DSM._get_effective_messages = _safe_get_effective_messages  # type: ignore[assignment]

    from langchain.agents.middleware.summarization import SummarizationMiddleware as _SM
    _original_cutoff = _SM._find_safe_cutoff_point
    @staticmethod  # type: ignore[misc]
    def _safe_cutoff_point(messages, cutoff_index):
        result = _original_cutoff(messages, cutoff_index)
        return min(result, max(0, len(messages) - 1))
    _SM._find_safe_cutoff_point = _safe_cutoff_point  # type: ignore[assignment]

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = _build_model(llm_provider, model_id, api_key, base_url)
    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)
    _tools_module._STORE = store

    from gagc.state import ThompsonState, ArmState
    from gagc.tools import _active_arms
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    LocalShellBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[
            propose_action_group,
            execute_trial_group,
            execute_trial,
            promote_winner,
            update_thompson_state,
            evaluate_final_incumbent,
        ],
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
    )

    return agent
