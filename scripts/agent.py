#!/usr/bin/env python3
"""RecClaw candidate orchestration agent.

Execution chain:
Observe -> Plan -> Act -> Evaluate -> Reflect -> Remember
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .collect_result import load_result_source
    from .compare_runs import compare_results
except ImportError:
    from collect_result import load_result_source
    from compare_runs import compare_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
NOTES_DIR = PROJECT_ROOT / "notes"

REQUIRED_MULTI_METRICS = (
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
    "latency_ms",
)
DEFAULT_METRIC_DIRECTIONS: dict[str, int] = {
    "run_time": -1,
    "latency_ms": -1,
}
EVALUATION_DIMENSIONS = (
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
    "latency_ms",
    "run_time",
)

PRIORITY_SCORE = {"high": 1.0, "medium": 0.5, "low": 0.2}
STATUS_SCORE = {"implemented": 1.0, "implement-ready": 0.7, "spec-ready": 0.3, "idea": 0.1}


@dataclass
class AgentConfig:
    rounds: int = 5
    metric: str = "ndcg@10"
    multi_metrics: bool = False
    min_keep_delta: float = 1e-5
    revisit_penalty: float = 0.1
    crash_penalty: float = 0.2
    novelty_weight: float = 0.45
    priority_weight: float = 0.25
    status_weight: float = 0.20
    recent_schedule_penalty: float = 0.15
    seed: int = 42
    dry_run: bool = False
    memory_read_limit: int = 2000
    memory_path: Path = MEMORY_PATH
    results_csv: Path = RESULTS_CSV
    baseline_dir: Path = BASELINE_DIR
    registry_path: Path = REGISTRY_PATH
    metrics_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ndcg@10": 0.2,
            "recall@10": 0.2,
            "mrr@10": 0.15,
            "hit@10": 0.15,
            "precision@10": 0.1,
            "itemcoverage@10": 0.1,
            "latency_ms": 0.1,
        }
    )
    metric_directions: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_METRIC_DIRECTIONS))
    parameter_space: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "embedding_size": [32, 64, 128],
            "learning_rate": [0.0001, 0.001, 0.005],
            "n_layers": [1, 2, 3],
            "reg_weight": [1e-6, 1e-5, 1e-4],
            "margin": [0.1, 0.2, 0.5],
            "residual_weight": [0.05, 0.1, 0.2, 0.3],
            "tail_weight_alpha": [0.2, 0.5, 1.0],
        }
    )


@dataclass
class TrialRecord:
    round_id: int
    candidate_id: str
    params: dict[str, Any]
    run_id: str
    status: str
    result: dict[str, Any]
    compare_baseline: dict[str, Any]
    compare_history_best: dict[str, Any]
    dimension_report: dict[str, Any]
    decision: str
    reason: str
    next_action: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_weights(weights_json: str) -> dict[str, float]:
    raw = json.loads(weights_json)
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("--metrics-weights-json must be an object")
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        metric = str(key).lower().strip()
        if metric not in REQUIRED_MULTI_METRICS:
            raise argparse.ArgumentTypeError(f"unsupported metric in weights: {key}")
        normalized[metric] = float(value)
    total = sum(normalized.values())
    if total <= 0:
        raise argparse.ArgumentTypeError("sum(weights) must be > 0")
    return {k: v / total for k, v in normalized.items()}


class RecClawAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.memory: list[dict[str, Any]] = []
        self.registry: list[dict[str, Any]] = []
        self.baselines_by_model: dict[str, dict[str, Any]] = {}
        self.history_by_candidate: dict[str, list[dict[str, Any]]] = {}
        self.best_by_model: dict[str, dict[str, Any]] = {}
        self.results_rows: list[dict[str, Any]] = []

    # ===== Observe =====
    def observe(self) -> None:
        payload = load_yaml(self.config.registry_path)
        self.registry = payload.get("candidates", [])
        if not isinstance(self.registry, list):
            raise ValueError("registry candidates must be a list")

        if self.config.memory_path.exists():
            lines = self.config.memory_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if self.config.memory_read_limit > 0:
                lines = lines[-self.config.memory_read_limit :]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                self.memory.append(json.loads(line))

        if self.config.results_csv.exists():
            with self.config.results_csv.open("r", encoding="utf-8", newline="") as handle:
                self.results_rows = list(csv.DictReader(handle))

        for item in self.memory:
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                self.history_by_candidate.setdefault(candidate_id, []).append(item)

        for log_file in self.config.baseline_dir.glob("*.log"):
            parsed = load_result_source(log_file)
            model = str(parsed.get("model") or "").strip()
            if model:
                if model in self.baselines_by_model:
                    print(f"Warning: duplicate baseline for model '{model}', overwriting with {log_file.name}")
                self.baselines_by_model[model] = parsed

        for row in self.results_rows:
            model = str(row.get("model") or "").strip()
            status = str(row.get("status") or "").lower()
            if not model or status != "success":
                continue
            if model not in self.best_by_model:
                self.best_by_model[model] = dict(row)
                continue
            if self._score(row) > self._score(self.best_by_model[model]):
                self.best_by_model[model] = dict(row)

    def _load_optional_context(self, candidate: dict[str, Any], force: bool = False) -> dict[str, str]:
        context: dict[str, str] = {}
        if not force:
            return context
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            return context
        cid_pattern = re.compile(rf"\b{re.escape(cid)}\b", flags=re.IGNORECASE)
        for name in ("candidate_library.md", "method_change_space.md"):
            path = NOTES_DIR / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            match = cid_pattern.search(text)
            if match is None:
                continue
            start = max(0, match.start() - 600)
            end = min(len(text), match.end() + 1200)
            context[name] = text[start:end]
        return context

    @staticmethod
    def _params_signature(params: dict[str, Any]) -> str:
        if not params:
            return "{}"
        return json.dumps(params, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _used_param_signatures(self, candidate_id: str) -> set[str]:
        signatures: set[str] = set()
        for row in self.history_by_candidate.get(candidate_id, []):
            params = row.get("params")
            if isinstance(params, dict):
                signatures.add(self._params_signature(params))
        return signatures

    def _choose_params(self, candidate: dict[str, Any]) -> dict[str, Any]:
        consumes = candidate.get("consumes") or []
        param_values: list[tuple[str, list[Any]]] = []
        params: dict[str, Any] = {}
        for key in consumes:
            key_text = str(key)
            values = self.config.parameter_space.get(key_text, [])
            if values:
                param_values.append((key_text, values))
                params[key_text] = self.rng.choice(values)

        if not param_values:
            return params

        used = self._used_param_signatures(str(candidate.get("candidate_id") or ""))
        current_sig = self._params_signature(params)
        if current_sig not in used:
            return params

        max_tries = max(20, 4 * len(param_values))
        for _ in range(max_tries):
            proposal = {k: self.rng.choice(v) for k, v in param_values}
            if self._params_signature(proposal) not in used:
                return proposal
        return params

    # ===== Plan =====
    def plan(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        runnable = [
            item
            for item in self.registry
            if bool(item.get("wired")) and str(item.get("runner_type") or "") in {"config_only", "model"}
        ]
        if not runnable:
            raise RuntimeError("no runnable candidates found in registry")

        best_score = float("-inf")
        chosen = runnable[0]
        for candidate in runnable:
            cid = str(candidate.get("candidate_id") or "")
            runs = self.history_by_candidate.get(cid, [])
            explored = len(runs)
            crashes = sum(1 for r in runs if str(r.get("decision")) == "crash")
            recently_scheduled = bool(self.memory) and str(self.memory[-1].get("candidate_id") or "") == cid
            priority = PRIORITY_SCORE.get(str(candidate.get("priority", "")).lower(), 0.0)
            status = STATUS_SCORE.get(str(candidate.get("status", "")).lower(), 0.0)
            novelty = 1.0 / (1.0 + explored)
            score = (
                self.config.novelty_weight * novelty
                + self.config.priority_weight * priority
                + self.config.status_weight * status
                - self.config.revisit_penalty * explored
                - self.config.crash_penalty * crashes
                - (self.config.recent_schedule_penalty if recently_scheduled else 0.0)
            )
            if score > best_score:
                best_score = score
                chosen = candidate

        params = self._choose_params(chosen)

        need_context = not self.history_by_candidate.get(str(chosen.get("candidate_id") or ""))
        context = self._load_optional_context(chosen, force=need_context)
        return chosen, params, context

    # ===== Act =====
    def act(self, candidate: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id"))
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_candidate.py"), candidate_id]
        for key, value in params.items():
            cmd.extend(["--set", f"{key}={value}"])

        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        summary = self._extract_last_json(completed.stdout)
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "summary": summary,
        }

    @staticmethod
    def _extract_last_json(text: str) -> dict[str, Any]:
        lines = [line for line in text.splitlines() if line.strip()]
        for idx in range(len(lines) - 1, -1, -1):
            chunk = "\n".join(lines[idx:])
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    # ===== Evaluate =====
    def evaluate(self, candidate: dict[str, Any], action_out: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        summary = action_out.get("summary", {})
        run_id = str(summary.get("run_id") or "")
        model_name = str(summary.get("model") or candidate.get("base_model") or "")
        if "/" in model_name:
            model_name = model_name.split("/", 1)[0].strip()

        candidate_result = {}
        result_json_path = summary.get("result_json_path")
        if result_json_path:
            candidate_result = load_result_source(result_json_path)
        elif run_id:
            candidate_result = self._load_result_from_csv(run_id)

        baseline = self.baselines_by_model.get(model_name, {})
        history_best = self.best_by_model.get(model_name, baseline)
        if not baseline:
            baseline = {"status": "missing"}
        if not history_best:
            history_best = baseline

        compare_baseline = compare_results(
            baseline,
            candidate_result,
            self.config.metric,
            multi_metrics=self.config.multi_metrics,
            metrics_weights=self.config.metrics_weights,
            metric_directions=self.config.metric_directions,
        )
        compare_history = compare_results(
            history_best,
            candidate_result,
            self.config.metric,
            multi_metrics=self.config.multi_metrics,
            metrics_weights=self.config.metrics_weights,
            metric_directions=self.config.metric_directions,
        )
        compare_baseline = self._normalize_compare_payload(compare_baseline)
        compare_history = self._normalize_compare_payload(compare_history)
        dimension_report = self._build_dimension_report(baseline, history_best, candidate_result)
        return candidate_result, compare_baseline, compare_history, dimension_report

    @staticmethod
    def _normalize_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        normalized.setdefault("delta", None)
        normalized.setdefault("decision", "crash")
        normalized.setdefault("explanation", "invalid comparison payload")
        return normalized

    def _load_result_from_csv(self, run_id: str) -> dict[str, Any]:
        for row in reversed(self.results_rows):
            if row.get("run_id") == run_id:
                return dict(row)
        return {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_dimension_report(
        self,
        baseline: dict[str, Any],
        history_best: dict[str, Any],
        candidate_result: dict[str, Any],
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for metric in EVALUATION_DIMENSIONS:
            cand = self._to_float(candidate_result.get(metric))
            base = self._to_float(baseline.get(metric))
            hist = self._to_float(history_best.get(metric))
            direction = -1 if int(self.config.metric_directions.get(metric, 1)) < 0 else 1
            report[metric] = {
                "candidate": cand,
                "baseline": base,
                "history_best": hist,
                "delta_vs_baseline": None if cand is None or base is None else round(cand - base, 6),
                "delta_vs_history_best": None if cand is None or hist is None else round(cand - hist, 6),
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
            }
        return report

    # ===== Reflect =====
    def reflect(
        self,
        candidate: dict[str, Any],
        params: dict[str, Any],
        action_out: dict[str, Any],
        candidate_result: dict[str, Any],
        compare_baseline: dict[str, Any],
        compare_history: dict[str, Any],
    ) -> tuple[str, str, str]:
        status = str(candidate_result.get("status") or "").lower()
        if action_out.get("exit_code", 1) != 0 or status != "success":
            return "crash", "candidate run crashed or result status is not success", "check candidate wiring and dependencies"

        baseline_delta = float(compare_baseline.get("delta") or 0.0)
        history_delta = float(compare_history.get("delta") or 0.0)
        if baseline_delta > self.config.min_keep_delta and history_delta > self.config.min_keep_delta:
            return "keep", "candidate improves both baseline and current history best", "schedule follow-up around this candidate"
        if baseline_delta >= 0 and history_delta <= self.config.min_keep_delta:
            return "revise", "candidate is near baseline but does not beat history best", self._revise_suggestion(candidate, params)
        if -self.config.min_keep_delta < baseline_delta < 0:
            return "revise", "candidate is slightly below baseline and needs parameter refinement", self._revise_suggestion(candidate, params)
        return "discard", "candidate does not beat baseline under current settings", "de-prioritize this configuration and explore another candidate"

    def _revise_suggestion(self, candidate: dict[str, Any], params: dict[str, Any]) -> str:
        consumes = candidate.get("consumes") or []
        if consumes:
            key = str(consumes[0])
            current = params.get(key)
            values = self.config.parameter_space.get(key, [])
            if values and current in values:
                idx = values.index(current)
                neighbors = []
                if idx > 0:
                    neighbors.append(values[idx - 1])
                if idx + 1 < len(values):
                    neighbors.append(values[idx + 1])
                if neighbors:
                    return f"try {key} in {neighbors} (current={current})"
            return f"adjust {key} to another value in configured parameter_space"
        return "refine implementation details or move to a related candidate"

    # ===== Remember =====
    def remember(self, record: TrialRecord) -> None:
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "round_id": record.round_id,
            "candidate_id": record.candidate_id,
            "params": record.params,
            "run_id": record.run_id,
            "status": record.status,
            "result": record.result,
            "compare_baseline": record.compare_baseline,
            "compare_history_best": record.compare_history_best,
            "dimension_report": record.dimension_report,
            "decision": record.decision,
            "reason": record.reason,
            "next_action": record.next_action,
        }
        with self.config.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.memory.append(payload)
        self.history_by_candidate.setdefault(record.candidate_id, []).append(payload)

    def _score(self, item: dict[str, Any]) -> float:
        if self.config.multi_metrics:
            score = 0.0
            for metric, weight in self.config.metrics_weights.items():
                direction = -1.0 if int(self.config.metric_directions.get(metric, 1)) < 0 else 1.0
                score += float(item.get(metric) or 0.0) * weight * direction
            return score
        return float(item.get(self.config.metric) or 0.0)

    def run(self) -> None:
        self.observe()
        for round_id in range(1, self.config.rounds + 1):
            try:
                candidate, params, context = self.plan()
                candidate_id = str(candidate.get("candidate_id"))
                print(f"\n[Round {round_id}] candidate={candidate_id} params={json.dumps(params, ensure_ascii=True)}")
                if context:
                    print(f"[Round {round_id}] optional_context_loaded={list(context.keys())}")
                if self.config.dry_run:
                    continue

                action_out = self.act(candidate, params)
                candidate_result, compare_baseline, compare_history, dimension_report = self.evaluate(candidate, action_out)
                decision, reason, next_action = self.reflect(
                    candidate, params, action_out, candidate_result, compare_baseline, compare_history
                )
                run_id = str((action_out.get("summary") or {}).get("run_id") or "")
                record = TrialRecord(
                    round_id=round_id,
                    candidate_id=candidate_id,
                    params=params,
                    run_id=run_id,
                    status=str(candidate_result.get("status") or "unknown"),
                    result=candidate_result,
                    compare_baseline=compare_baseline,
                    compare_history_best=compare_history,
                    dimension_report=dimension_report,
                    decision=decision,
                    reason=reason,
                    next_action=next_action,
                )
                self.remember(record)
                print(
                    json.dumps(
                        {
                            "round": round_id,
                            "candidate_id": candidate_id,
                            "run_id": run_id,
                            "decision": decision,
                            "reason": reason,
                            "next_action": next_action,
                            "latency_ms": candidate_result.get("latency_ms", ""),
                        },
                        ensure_ascii=True,
                        indent=2,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                candidate_id = "unknown"
                run_id = f"agent_internal_round_{round_id}_{int(datetime.now().timestamp())}"
                reason = f"{type(exc).__name__}: {exc}"
                fallback_record = TrialRecord(
                    round_id=round_id,
                    candidate_id=candidate_id,
                    params={},
                    run_id=run_id,
                    status="crash",
                    result={},
                    compare_baseline={"delta": None, "decision": "crash", "explanation": reason},
                    compare_history_best={"delta": None, "decision": "crash", "explanation": reason},
                    dimension_report={},
                    decision="crash",
                    reason=reason,
                    next_action="inspect agent pipeline/runtime environment and retry",
                )
                self.remember(fallback_record)
                print(f"[Round {round_id}] crash: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RecClaw Observe->Plan->Act->Evaluate->Reflect->Remember agent")
    parser.add_argument("--rounds", type=int, default=5, help="Number of scheduling rounds")
    parser.add_argument("--metric", default="ndcg@10", help="Primary metric for single-objective mode")
    parser.add_argument("--multi-metrics", action="store_true", help="Enable weighted multi-metric scoring")
    parser.add_argument("--metrics-weights-json", help="Weights JSON for multi-metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run Observe+Plan only")
    parser.add_argument("--memory-path", default=str(MEMORY_PATH), help="Agent memory jsonl path")
    parser.add_argument("--memory-read-limit", type=int, default=2000, help="Read only latest N memory rows (0 means all)")
    parser.add_argument("--recent-schedule-penalty", type=float, default=0.15, help="Penalty if candidate was scheduled in previous round")
    parser.add_argument(
        "--metric-lower-is-better",
        action="append",
        default=[],
        help="Metric names with lower-is-better semantics; repeatable",
    )
    args = parser.parse_args()

    metrics_weights = AgentConfig().metrics_weights
    if args.metrics_weights_json:
        metrics_weights = normalize_weights(args.metrics_weights_json)

    config = AgentConfig(
        rounds=max(1, args.rounds),
        metric=args.metric,
        multi_metrics=args.multi_metrics,
        metrics_weights=metrics_weights,
        seed=args.seed,
        dry_run=args.dry_run,
        memory_path=Path(args.memory_path),
        memory_read_limit=max(0, args.memory_read_limit),
        recent_schedule_penalty=max(0.0, args.recent_schedule_penalty),
        metric_directions={
            **DEFAULT_METRIC_DIRECTIONS,
            **{str(name).lower(): -1 for name in args.metric_lower_is_better},
        },
    )
    RecClawAgent(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
