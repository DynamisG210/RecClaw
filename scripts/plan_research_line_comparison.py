#!/usr/bin/env python3
"""Plan paired RecClaw-vs-original comparison runs.

The script does not modify the original project. It writes a JSON command plan
under the current RecClaw-7.23 results directory so formal server experiments
can run both systems with matched budgets and isolated artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OLD_ROOT = PROJECT_ROOT.parent / "RecClaw-原版"
DEFAULT_OUT = PROJECT_ROOT / "results" / "comparisons" / "research_line_vs_original_plan.json"


def command_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def validate_project_root(path: Path) -> None:
    required = path / "scripts" / "run_reflection_pilot.py"
    if not required.exists():
        raise ValueError(f"project root is missing scripts/run_reflection_pilot.py: {path}")


def pilot_command(
    *,
    root: Path,
    pilot_root: Path,
    stamp: str,
    rounds: int,
    gpu_id: int,
    proposal_source: str,
    loop_mode: str,
    search_intensity: str,
    baseline_dir: str,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key_env: str,
    llm_timeout: int,
    llm_retries: int,
    llm_temperature: float,
    llm_max_tokens: int,
    proposal_count: int | None,
    proposal_every: int | None,
    allow_llm_fallback: bool,
    extra_overrides: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_reflection_pilot.py"),
        "--rounds",
        str(rounds),
        "--gpu-id",
        str(gpu_id),
        "--proposal-source",
        proposal_source,
        "--loop-mode",
        loop_mode,
        "--search-intensity",
        search_intensity,
        "--pilot-root",
        str(pilot_root),
        "--stamp",
        stamp,
    ]
    if proposal_count is not None:
        cmd.extend(["--proposal-count", str(proposal_count)])
    if proposal_every is not None:
        cmd.extend(["--proposal-every", str(proposal_every)])
    if baseline_dir:
        cmd.extend(["--baseline-dir", baseline_dir])
    if proposal_source == "llm":
        cmd.extend(["--llm-provider", llm_provider])
        if llm_model:
            cmd.extend(["--llm-model", llm_model])
        if llm_base_url:
            cmd.extend(["--llm-base-url", llm_base_url])
        if llm_api_key_env:
            cmd.extend(["--llm-api-key-env", llm_api_key_env])
        cmd.extend(["--llm-timeout", str(llm_timeout)])
        cmd.extend(["--llm-retries", str(llm_retries)])
        cmd.extend(["--llm-temperature", str(llm_temperature)])
        cmd.extend(["--llm-max-tokens", str(llm_max_tokens)])
        if allow_llm_fallback:
            cmd.append("--allow-llm-fallback")
    for override in extra_overrides:
        cmd.extend(["--set", override])
    return cmd


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    old_root = Path(args.old_root).expanduser().resolve()
    validate_project_root(PROJECT_ROOT)
    validate_project_root(old_root)
    comparison_root = Path(args.comparison_root).expanduser().resolve()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = []
    for index in range(max(1, args.repeats)):
        pair_stamp = f"{stamp}_rep{index + 1:02d}"
        new_cmd = pilot_command(
            root=PROJECT_ROOT,
            pilot_root=comparison_root / "research_line",
            stamp=pair_stamp,
            rounds=max(1, args.rounds),
            gpu_id=args.gpu_id,
            proposal_source=args.proposal_source,
            loop_mode=args.new_loop_mode,
            search_intensity=args.new_search_intensity,
            baseline_dir=args.baseline_dir,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            llm_timeout=max(1, args.llm_timeout),
            llm_retries=max(0, args.llm_retries),
            llm_temperature=float(args.llm_temperature),
            llm_max_tokens=max(1, args.llm_max_tokens),
            proposal_count=args.proposal_count,
            proposal_every=args.proposal_every,
            allow_llm_fallback=args.allow_llm_fallback,
            extra_overrides=list(args.extra_overrides or []),
        )
        old_cmd = pilot_command(
            root=old_root,
            pilot_root=comparison_root / "original",
            stamp=pair_stamp,
            rounds=max(1, args.rounds),
            gpu_id=args.gpu_id,
            proposal_source=args.proposal_source,
            loop_mode=args.old_loop_mode,
            search_intensity=args.old_search_intensity,
            baseline_dir=args.baseline_dir,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            llm_timeout=max(1, args.llm_timeout),
            llm_retries=max(0, args.llm_retries),
            llm_temperature=float(args.llm_temperature),
            llm_max_tokens=max(1, args.llm_max_tokens),
            proposal_count=args.proposal_count,
            proposal_every=args.proposal_every,
            allow_llm_fallback=args.allow_llm_fallback,
            extra_overrides=list(args.extra_overrides or []),
        )
        pairs.append(
            {
                "replicate": index + 1,
                "stamp": pair_stamp,
                "research_line": {
                    "project_root": str(PROJECT_ROOT),
                    "run_dir": str(comparison_root / "research_line" / pair_stamp),
                    "command": new_cmd,
                    "command_text": command_text(new_cmd),
                },
                "original": {
                    "project_root": str(old_root),
                    "run_dir": str(comparison_root / "original" / pair_stamp),
                    "command": old_cmd,
                    "command_text": command_text(old_cmd),
                },
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "compare RecClaw research-ability line against original RecClaw",
        "fairness_controls": {
            "rounds": max(1, args.rounds),
            "repeats": max(1, args.repeats),
            "gpu_id": args.gpu_id,
            "proposal_source": args.proposal_source,
            "baseline_dir": args.baseline_dir,
            "proposal_count": args.proposal_count,
            "proposal_every": args.proposal_every,
            "llm_timeout": max(1, args.llm_timeout),
            "llm_retries": max(0, args.llm_retries),
            "llm_temperature": float(args.llm_temperature),
            "llm_max_tokens": max(1, args.llm_max_tokens),
            "extra_overrides": list(args.extra_overrides or []),
            "same_protocol_requirement": "Both runs must keep ML-1M split/evaluator/metric unchanged.",
            "primary_metric": "ndcg@10",
        },
        "treatments": {
            "research_line": {
                "loop_mode": args.new_loop_mode,
                "search_intensity": args.new_search_intensity,
                "expected_enabled_features": [
                    "explicit Candidate Producers",
                    "Research Router",
                    "Search Memory",
                    "Meta-Research advisory trace",
                ],
            },
            "original": {
                "loop_mode": args.old_loop_mode,
                "search_intensity": args.old_search_intensity,
                "expected_enabled_features": ["original proposal/planning behavior"],
            },
        },
        "pairs": pairs,
        "analysis_notes": [
            "Compare best ndcg@10 by round budget, keep/revise/discard/crash rates, duplicate proposal rate, and code_required implementation success rate.",
            "Do not mix evidence across protocol-changing runs; discard any run that changes split, evaluation mode, candidate universe, or metric semantics.",
            "Run commands in pair order or randomize pair order externally if GPU contention is a concern.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a paired research-line vs original RecClaw experiment.")
    parser.add_argument("--old-root", default=str(DEFAULT_OLD_ROOT), help="Path to original RecClaw project")
    parser.add_argument("--comparison-root", default=str(PROJECT_ROOT / "results" / "comparisons" / "runs"))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON plan path")
    parser.add_argument("--stamp", default="", help="Optional deterministic stamp prefix")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--proposal-source", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--baseline-dir", default="", help="Optional shared same-protocol baseline dir")
    parser.add_argument("--new-loop-mode", choices=("tuning", "mixed", "explore", "auto"), default="auto")
    parser.add_argument("--new-search-intensity", choices=("balanced", "algorithm_first"), default="algorithm_first")
    parser.add_argument("--old-loop-mode", choices=("tuning", "mixed", "explore", "auto"), default="mixed")
    parser.add_argument("--old-search-intensity", choices=("balanced", "algorithm_first"), default="balanced")
    parser.add_argument("--llm-provider", choices=("deepseek", "openai", "compatible"), default="deepseek")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key-env", default="")
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--proposal-count", type=int, default=None)
    parser.add_argument("--proposal-every", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="extra_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional RecBole override forwarded to both planned runs; omit for full formal training.",
    )
    parser.add_argument("--allow-llm-fallback", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(args)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "pairs": len(plan["pairs"])}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
