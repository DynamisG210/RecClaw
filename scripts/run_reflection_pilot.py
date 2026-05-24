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
    return bool(env.get("DEEPSEEK_API_KEY") or env.get("OPENAI_API_KEY"))


def choose_proposal_source(explicit: str | None, env: dict[str, str] | None = None) -> str:
    if explicit:
        return explicit
    return "llm" if has_llm_key(env) else "heuristic"


def reject_lablog_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "recclaw_lablog" in text:
        raise ValueError(f"pilot runtime refuses LabLog path: {path}")


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
    refresh_every: int,
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
        "auto",
        "--proposal-source",
        proposal_source,
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
    if proposal_source == "llm":
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


def plan_payload(paths: PilotPaths, commands: dict[str, list[str]], proposal_source: str, gpu_id: int) -> dict[str, Any]:
    return {
        "run_dir": str(paths.run_dir),
        "proposal_source": proposal_source,
        "gpu_id": gpu_id,
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
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT), help="Root directory for isolated pilot runs")
    parser.add_argument("--stamp", default="", help="Optional deterministic run directory suffix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and paths without running them")
    args = parser.parse_args(argv)

    root = Path(args.pilot_root).expanduser().resolve()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = build_paths(root, stamp)
    for value in paths.__dict__.values():
        reject_lablog_path(value)
    proposal_source = choose_proposal_source(args.proposal_source)
    commands = build_commands(
        paths=paths,
        rounds=max(1, args.rounds),
        gpu_id=int(args.gpu_id),
        proposal_source=proposal_source,
        refresh_every=max(1, args.refresh_experience_every),
    )

    payload = plan_payload(paths, commands, proposal_source, int(args.gpu_id))
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    ensure_runtime_dirs(paths)
    env = build_env(paths)
    for name in ("lint", "initial_tree", "initial_summary", "agent"):
        run_checked(commands[name], env=env)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
