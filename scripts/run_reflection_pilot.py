#!/usr/bin/env python3
"""Run an isolated full-memory reflection pilot.

The pilot keeps runtime artifacts under results/pilots/<timestamp>/ and uses
the same RecBole evaluation protocol as the normal candidate runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "results" / "pilots"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
SAFE_OVERRIDES = (
    "train_batch_size=2048",
    "eval_batch_size=65536",
    "worker=8",
)


@dataclass(frozen=True)
class PilotPaths:
    run_dir: Path
    memory: Path
    state_summary: Path
    proposals: Path
    results_csv: Path
    candidate_result_dir: Path
    override_dir: Path
    checkpoint_dir: Path
    candidate_tree_json: Path
    candidate_tree_md: Path
    candidate_tree_mmd: Path
    experience_summary_md: Path
    experience_summary_json: Path
    reflection_memory: Path


def has_llm_key(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    custom_key_env = env.get("RECCLAW_LLM_API_KEY_ENV", "")
    return bool(
        env.get("DEEPSEEK_API_KEY")
        or env.get("OPENAI_API_KEY")
        or (custom_key_env and env.get(custom_key_env))
    )


def choose_proposal_source(explicit: str | None, env: dict[str, str] | None = None) -> str:
    if explicit:
        return explicit
    return "llm" if has_llm_key(env) else "heuristic"


def choose_loop_mode(explicit: str | None, proposal_source: str) -> str:
    if explicit:
        return explicit
    return "auto" if proposal_source == "llm" else "mixed"


def choose_llm_provider(explicit: str | None, env: dict[str, str] | None = None) -> str:
    if explicit:
        return explicit
    env = env or os.environ
    if env.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    return "deepseek"


def default_llm_key_env(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    return "RECCLAW_LLM_API_KEY"


def validate_llm_env(*, proposal_source: str, key_env: str, env: dict[str, str] | None = None) -> None:
    if proposal_source != "llm":
        return
    env = env or os.environ
    if not env.get(key_env, ""):
        raise ValueError(
            f"proposal-source=llm requires {key_env} in the server environment; "
            "use --proposal-source heuristic for a no-key pilot"
        )


def reject_lablog_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "recclaw_lablog" in text:
        raise ValueError(f"pilot runtime refuses LabLog path: {path}")


def resolve_baseline_dir(explicit: str | None, env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    raw = explicit or env.get("RECCLAW_BASELINE_DIR") or str(DEFAULT_BASELINE_DIR)
    return Path(raw).expanduser().resolve()


def validate_agent_baseline_dir(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"baseline dir does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"baseline path is not a directory: {path}")
    if not any(path.glob("*.log")):
        raise ValueError(
            "baseline dir must contain top-level .log files because agent.py reads "
            f"baseline_dir.glob('*.log'): {path}"
        )


def build_paths(root: Path, stamp: str) -> PilotPaths:
    run_dir = root / stamp
    return PilotPaths(
        run_dir=run_dir,
        memory=run_dir / "agent_memory.jsonl",
        state_summary=run_dir / "agent_state_summary.json",
        proposals=run_dir / "candidate_proposals.jsonl",
        results_csv=run_dir / "results.csv",
        candidate_result_dir=run_dir / "candidates",
        override_dir=run_dir / "overrides",
        checkpoint_dir=run_dir / "checkpoints",
        candidate_tree_json=run_dir / "candidate_search_tree.json",
        candidate_tree_md=run_dir / "candidate_search_tree.md",
        candidate_tree_mmd=run_dir / "candidate_search_tree.mmd",
        experience_summary_md=run_dir / "experience_summary.md",
        experience_summary_json=run_dir / "experience_summary.json",
        reflection_memory=run_dir / "reflection_memory.jsonl",
    )


def ensure_runtime_dirs(paths: PilotPaths) -> None:
    for path in (
        paths.run_dir,
        paths.candidate_result_dir,
        paths.override_dir,
        paths.checkpoint_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def build_commands(
    *,
    paths: PilotPaths,
    rounds: int,
    gpu_id: int,
    proposal_source: str,
    loop_mode: str,
    llm_provider: str,
    llm_api_key_env: str,
    llm_model: str,
    llm_base_url: str,
    refresh_every: int,
    baseline_dir: Path,
) -> dict[str, list[str]]:
    lint_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analysis" / "lint_recclaw_space.py"),
    ]
    tree_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analysis" / "build_candidate_search_tree.py"),
        "--memory",
        str(paths.memory),
        "--results",
        str(paths.results_csv),
        "--proposals",
        str(paths.proposals),
        "--out-json",
        str(paths.candidate_tree_json),
        "--out-md",
        str(paths.candidate_tree_md),
        "--out-mmd",
        str(paths.candidate_tree_mmd),
    ]
    summary_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_experience_summary.py"),
        "--memory",
        str(paths.memory),
        "--results",
        str(paths.results_csv),
        "--tree",
        str(paths.candidate_tree_json),
        "--out-md",
        str(paths.experience_summary_md),
        "--out-json",
        str(paths.experience_summary_json),
        "--out-jsonl",
        str(paths.reflection_memory),
    ]
    agent_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "agent.py"),
        "--rounds",
        str(rounds),
        "--loop-mode",
        loop_mode,
        "--proposal-source",
        proposal_source,
        "--llm-provider",
        llm_provider,
        "--llm-api-key-env",
        llm_api_key_env,
        "--proposal-count",
        "5",
        "--proposal-every",
        "3",
        "--memory-path",
        str(paths.memory),
        "--state-summary-path",
        str(paths.state_summary),
        "--proposal-path",
        str(paths.proposals),
        "--results-csv",
        str(paths.results_csv),
        "--baseline-dir",
        str(baseline_dir),
        "--experiment-directive-file",
        str(paths.experience_summary_md),
        "--experiment-directive",
        (
            "Reflection pilot constraints: keep the ML-1M full-sort general-rec protocol unchanged; "
            "do not propose sequential, posthoc, SSL, diversity, or planned-parameter mechanisms; "
            "prioritize wired registry candidates and tuning around the first-batch BPR/LightGCN candidates."
        ),
        "--refresh-experience-every",
        str(refresh_every),
        "--candidate-tree-path",
        str(paths.candidate_tree_json),
        "--candidate-tree-md-path",
        str(paths.candidate_tree_md),
        "--candidate-tree-mmd-path",
        str(paths.candidate_tree_mmd),
        "--experience-summary-path",
        str(paths.experience_summary_md),
        "--experience-summary-json-path",
        str(paths.experience_summary_json),
        "--reflection-memory-path",
        str(paths.reflection_memory),
        "--checkpoint-dir",
        str(paths.checkpoint_dir),
        "--checkpoint-policy",
        "cleanup_all",
    ]
    for override in (*SAFE_OVERRIDES, f"gpu_id={gpu_id}"):
        agent_cmd.extend(["--set", override])
    if llm_model:
        agent_cmd.extend(["--llm-model", llm_model])
    if llm_base_url:
        agent_cmd.extend(["--llm-base-url", llm_base_url])
    if proposal_source == "llm" or loop_mode == "auto":
        agent_cmd.append("--allow-llm-fallback")
    return {
        "lint": lint_cmd,
        "initial_tree": tree_cmd,
        "initial_summary": summary_cmd,
        "agent": agent_cmd,
    }


def build_env(paths: PilotPaths) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RECCLAW_RESULT_DIR": str(paths.candidate_result_dir),
            "RECCLAW_RESULTS_CSV": str(paths.results_csv),
            "RECCLAW_OVERRIDE_DIR": str(paths.override_dir),
            "RECCLAW_CHECKPOINT_DIR": str(paths.checkpoint_dir),
            "RECCLAW_CHECKPOINT_POLICY": "cleanup_all",
        }
    )
    return env


def run_checked(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[pilot] $ {shlex.join(cmd)}")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def plan_payload(
    paths: PilotPaths,
    commands: dict[str, list[str]],
    proposal_source: str,
    loop_mode: str,
    llm_provider: str,
    llm_api_key_env: str,
    gpu_id: int,
    baseline_dir: Path,
) -> dict[str, Any]:
    return {
        "run_dir": str(paths.run_dir),
        "proposal_source": proposal_source,
        "loop_mode": loop_mode,
        "llm_provider": llm_provider,
        "llm_api_key_env": llm_api_key_env,
        "gpu_id": gpu_id,
        "baseline_dir": str(baseline_dir),
        "safe_overrides": list(SAFE_OVERRIDES),
        "commands": {name: shlex.join(cmd) for name, cmd in commands.items()},
        "env": {
            "RECCLAW_RESULT_DIR": str(paths.candidate_result_dir),
            "RECCLAW_RESULTS_CSV": str(paths.results_csv),
            "RECCLAW_OVERRIDE_DIR": str(paths.override_dir),
            "RECCLAW_CHECKPOINT_DIR": str(paths.checkpoint_dir),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a 50-round full-memory reflection pilot.")
    parser.add_argument("--gpu-id", type=int, default=0, help="RecBole gpu_id override for candidate runs")
    parser.add_argument("--rounds", type=int, default=50, help="Number of agent rounds")
    parser.add_argument("--refresh-experience-every", type=int, default=10, help="Refresh cadence in rounds")
    parser.add_argument("--proposal-source", choices=("llm", "heuristic"), default=None)
    parser.add_argument(
        "--loop-mode",
        choices=("tuning", "mixed", "explore", "auto"),
        default=None,
        help="Agent loop mode. Defaults to auto for LLM pilots and mixed for no-key heuristic pilots.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory containing same-protocol baseline logs/results for agent comparisons",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("deepseek", "openai", "compatible"),
        default=None,
        help="LLM provider passed through to agent.py. Defaults from available key env vars.",
    )
    parser.add_argument("--llm-model", default="", help="Optional LLM model passed through to agent.py")
    parser.add_argument("--llm-base-url", default="", help="Optional OpenAI-compatible API base URL")
    parser.add_argument("--llm-api-key-env", default="", help="Environment variable containing the LLM API key")
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT), help="Root directory for isolated pilot runs")
    parser.add_argument("--stamp", default="", help="Optional deterministic run directory suffix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and paths without running them")
    args = parser.parse_args(argv)

    root = Path(args.pilot_root).expanduser().resolve()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = build_paths(root, stamp)
    for value in paths.__dict__.values():
        reject_lablog_path(value)
    baseline_dir = resolve_baseline_dir(args.baseline_dir)
    reject_lablog_path(baseline_dir)
    proposal_source = choose_proposal_source(args.proposal_source)
    loop_mode = choose_loop_mode(args.loop_mode, proposal_source)
    llm_provider = choose_llm_provider(args.llm_provider)
    llm_api_key_env = default_llm_key_env(llm_provider, args.llm_api_key_env or None)
    commands = build_commands(
        paths=paths,
        rounds=max(1, args.rounds),
        gpu_id=int(args.gpu_id),
        proposal_source=proposal_source,
        loop_mode=loop_mode,
        llm_provider=llm_provider,
        llm_api_key_env=llm_api_key_env,
        llm_model=str(args.llm_model or ""),
        llm_base_url=str(args.llm_base_url or ""),
        refresh_every=max(1, args.refresh_experience_every),
        baseline_dir=baseline_dir,
    )

    payload = plan_payload(
        paths,
        commands,
        proposal_source,
        loop_mode,
        llm_provider,
        llm_api_key_env,
        int(args.gpu_id),
        baseline_dir,
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    try:
        validate_agent_baseline_dir(baseline_dir)
        validate_llm_env(proposal_source=proposal_source, key_env=llm_api_key_env)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ensure_runtime_dirs(paths)
    env = build_env(paths)
    for name in ("lint", "initial_tree", "initial_summary", "agent"):
        run_checked(commands[name], env=env)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
