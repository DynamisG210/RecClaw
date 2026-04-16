#!/usr/bin/env python3
"""RecClaw Agent for automatic recommendation algorithm research."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # Package import path (e.g. import scripts.agent)
    from .collect_result import load_result_source
    from .compare_runs import compare_results
except ImportError:
    # Script execution path (e.g. py scripts/agent.py)
    from collect_result import load_result_source
    from compare_runs import compare_results

REQUIRED_MULTI_METRICS = (
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
)


@dataclass
class AgentConfig:
    """Configuration for the RecClaw Agent."""
    models: List[str] = field(default_factory=lambda: ["BPR", "LightGCN"])
    embedding_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    learning_rates: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.005])
    n_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    reg_weights: List[float] = field(default_factory=lambda: [1e-6, 1e-5, 1e-4])
    max_trials: int = 10
    metric: str = "ndcg@10"
    multi_metrics: bool = False
    metrics_weights: dict[str, float] = field(default_factory=lambda: {
        "ndcg@10": 1.0 / 6.0,
        "recall@10": 1.0 / 6.0,
        "mrr@10": 1.0 / 6.0,
        "hit@10": 1.0 / 6.0,
        "precision@10": 1.0 / 6.0,
        "itemcoverage@10": 1.0 / 6.0,
    })
    baseline_dir: str = "results/baseline"
    candidate_dir: str = "results/candidates"
    results_csv: str = "results/results.csv"


@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    run_id: str
    model: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str
    run_time: float


class RecClawAgent:
    """RecClaw Agent for automatic recommendation algorithm research."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent with the given configuration."""
        self.config = config or AgentConfig()
        self.baseline_results: Dict[str, ExperimentResult] = {}
        self.candidate_results: List[ExperimentResult] = []
        self.best_result: Optional[ExperimentResult] = None
        self.best_score: float = float("-inf")

    def generate_config(self, model: str) -> Dict[str, Any]:
        """Generate a random configuration for the given model."""
        config = {}
        
        # Common parameters
        config["embedding_size"] = random.choice(self.config.embedding_sizes)
        config["learning_rate"] = random.choice(self.config.learning_rates)
        
        # Model-specific parameters
        if model == "LightGCN":
            config["n_layers"] = random.choice(self.config.n_layers)
            config["reg_weight"] = random.choice(self.config.reg_weights)
        elif model == "BPR":
            config["reg_weight"] = random.choice(self.config.reg_weights)
        
        return config

    def run_experiment(self, model: str, config: Dict[str, Any]) -> ExperimentResult:
        """Run an experiment with the given model and configuration."""
        run_id = f"{model}_{int(time.time())}_{random.randint(1000, 9999)}"
        log_path = f"{self.config.candidate_dir}/{run_id}.log"
        
        # Create candidate directory if it doesn't exist
        Path(self.config.candidate_dir).mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "py", "-3.10", "d:\\Apex\\Recclaw\\RecBole-master\\run_recbole.py",
            "--model", model,
            "--dataset", "ml-1m",
            "--config_files", f"configs/task_ml1m.yaml configs/{model.lower()}.yaml",
            "--embedding_size", str(config["embedding_size"]),
            "--learning_rate", str(config["learning_rate"]),
            "--nproc", "1",
        ]
        
        if model == "LightGCN":
            cmd.extend([
                "--n_layers", str(config["n_layers"]),
                "--reg_weight", str(config["reg_weight"]),
            ])
        elif model == "BPR":
            cmd.extend(["--reg_weight", str(config["reg_weight"])])
        
        # Run the experiment
        print(f"Running experiment: {run_id}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            with open(log_path, "w") as f:
                subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=Path(__file__).parent.parent
                )
            status = "success"
        except subprocess.CalledProcessError:
            status = "crash"
        
        # Parse the result
        result = load_result_source(log_path)
        metrics = {
            "ndcg@10": result.get("ndcg@10", 0.0),
            "recall@10": result.get("recall@10", 0.0),
            "mrr@10": result.get("mrr@10", 0.0),
            "hit@10": result.get("hit@10", 0.0),
            "precision@10": result.get("precision@10", 0.0),
            "itemcoverage@10": result.get("itemcoverage@10", 0.0),
        }
        
        return ExperimentResult(
            run_id=run_id,
            model=model,
            config=config,
            metrics=metrics,
            status=status,
            run_time=result.get("run_time", 0.0)
        )

    def load_baselines(self) -> None:
        """Load baseline results."""
        baseline_dir = Path(self.config.baseline_dir)
        if not baseline_dir.exists():
            print(f"Baseline directory {baseline_dir} does not exist.")
            return
        
        for log_file in baseline_dir.glob("*.log"):
            try:
                result = load_result_source(log_file)
                model = result.get("model", "")
                if model:
                    self.baseline_results[model] = ExperimentResult(
                        run_id=log_file.stem,
                        model=model,
                        config={},
                        metrics={
                            "ndcg@10": result.get("ndcg@10", 0.0),
                            "recall@10": result.get("recall@10", 0.0),
                            "mrr@10": result.get("mrr@10", 0.0),
                            "hit@10": result.get("hit@10", 0.0),
                            "precision@10": result.get("precision@10", 0.0),
                            "itemcoverage@10": result.get("itemcoverage@10", 0.0),
                        },
                        status=result.get("status", ""),
                        run_time=result.get("run_time", 0.0)
                    )
            except Exception as e:
                print(f"Error loading baseline {log_file}: {e}")

    def compare_with_baseline(self, candidate: ExperimentResult) -> Dict[str, Any]:
        """Compare a candidate result with the baseline."""
        baseline = self.baseline_results.get(candidate.model)
        if not baseline:
            return {
                "decision": "crash",
                "explanation": f"No baseline found for model {candidate.model}"
            }
        
        # Build baseline and candidate dictionaries with all metrics
        baseline_dict = {
            "ndcg@10": baseline.metrics["ndcg@10"],
            "recall@10": baseline.metrics["recall@10"],
            "mrr@10": baseline.metrics.get("mrr@10", 0.0),
            "hit@10": baseline.metrics.get("hit@10", 0.0),
            "precision@10": baseline.metrics.get("precision@10", 0.0),
            "itemcoverage@10": baseline.metrics.get("itemcoverage@10", 0.0),
            "status": baseline.status
        }
        
        candidate_dict = {
            "ndcg@10": candidate.metrics["ndcg@10"],
            "recall@10": candidate.metrics["recall@10"],
            "mrr@10": candidate.metrics.get("mrr@10", 0.0),
            "hit@10": candidate.metrics.get("hit@10", 0.0),
            "precision@10": candidate.metrics.get("precision@10", 0.0),
            "itemcoverage@10": candidate.metrics.get("itemcoverage@10", 0.0),
            "status": candidate.status
        }
        
        # Use multi-objective optimization if enabled
        if self.config.multi_metrics:
            return compare_results(
                baseline_dict, 
                candidate_dict, 
                self.config.metric,
                multi_metrics=True,
                metrics_weights=self.config.metrics_weights
            )
        else:
            # Use single metric comparison (original behavior)
            return compare_results(baseline_dict, candidate_dict, self.config.metric)

    def score_result(self, result: ExperimentResult) -> float:
        """Score a result with current optimization mode."""
        if self.config.multi_metrics:
            score = 0.0
            for metric, weight in self.config.metrics_weights.items():
                score += result.metrics.get(metric, 0.0) * weight
            return score
        return result.metrics.get(self.config.metric, 0.0)

    def run(self) -> None:
        """Run the agent to explore configurations and find the best one."""
        # Load baselines
        self.load_baselines()
        
        # Run trials
        for i in range(self.config.max_trials):
            print(f"\nTrial {i+1}/{self.config.max_trials}")
            
            # Select a model
            model = random.choice(self.config.models)
            
            # Generate configuration
            config = self.generate_config(model)
            
            # Run experiment
            result = self.run_experiment(model, config)
            self.candidate_results.append(result)
            
            # Compare with baseline
            comparison = self.compare_with_baseline(result)
            print(f"Comparison result: {comparison['decision']}")
            print(f"Explanation: {comparison['explanation']}")
            
            # Update best result
            if comparison['decision'] == "keep":
                current_score = self.score_result(result)
                if not self.best_result or current_score > self.best_score:
                    self.best_result = result
                    self.best_score = current_score
                    if self.config.multi_metrics:
                        print(f"New best result found: {result.run_id} with weighted_score = {current_score:.6f}")
                    else:
                        print(f"New best result found: {result.run_id} with {self.config.metric} = {result.metrics[self.config.metric]}")
        
        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print a summary of the agent's results."""
        print("\n" + "="*80)
        print("RecClaw Agent Summary")
        print("="*80)
        
        print(f"Total trials: {len(self.candidate_results)}")
        print(f"Baselines loaded: {list(self.baseline_results.keys())}")
        
        if self.best_result:
            print("\nBest result:")
            print(f"  Run ID: {self.best_result.run_id}")
            print(f"  Model: {self.best_result.model}")
            print(f"  Configuration: {json.dumps(self.best_result.config, indent=2)}")
            print(f"  Metrics: {json.dumps(self.best_result.metrics, indent=2)}")
            if self.config.multi_metrics:
                print(f"  Weighted score: {self.best_score:.6f}")
            print(f"  Run time: {self.best_result.run_time} seconds")
        else:
            print("\nNo best result found.")
        
        print("="*80)


def main() -> int:
    """Main function to run the agent."""
    import argparse

    def parse_metrics_weights(weights_json: str) -> dict[str, float]:
        """Parse and validate metric weights JSON string."""
        try:
            raw_weights = json.loads(weights_json)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON for --metrics-weights-json: {exc}"
            ) from exc

        if not isinstance(raw_weights, dict):
            raise argparse.ArgumentTypeError("--metrics-weights-json must be a JSON object")

        normalized_weights: dict[str, float] = {}
        for key, value in raw_weights.items():
            metric = str(key).lower().strip()
            if metric not in REQUIRED_MULTI_METRICS:
                raise argparse.ArgumentTypeError(
                    f"Unsupported metric in weights: {key}. Allowed: {', '.join(REQUIRED_MULTI_METRICS)}"
                )
            try:
                normalized_weights[metric] = float(value)
            except (TypeError, ValueError) as exc:
                raise argparse.ArgumentTypeError(
                    f"Weight for {key} must be numeric"
                ) from exc

        missing = [metric for metric in REQUIRED_MULTI_METRICS if metric not in normalized_weights]
        if missing:
            raise argparse.ArgumentTypeError(
                f"Missing weights for metrics: {', '.join(missing)}"
            )

        total = sum(normalized_weights.values())
        if total <= 0:
            raise argparse.ArgumentTypeError("Sum of metric weights must be > 0")

        # Normalize to 1.0 so users can provide either percentages or raw weights.
        return {metric: weight / total for metric, weight in normalized_weights.items()}

    parser = argparse.ArgumentParser(description="Run RecClaw Agent")
    parser.add_argument("--multi-metrics", action="store_true", help="Enable multi-objective optimization")
    parser.add_argument("--metric", default="ndcg@10", help="Primary metric for single-objective optimization")
    parser.add_argument(
        "--metrics-weights-json",
        type=parse_metrics_weights,
        help=(
            "Custom weights JSON for multi-metrics, e.g. "
            "'{\"ndcg@10\":0.3,\"recall@10\":0.2,\"mrr@10\":0.15,"
            "\"hit@10\":0.15,\"precision@10\":0.1,\"itemcoverage@10\":0.1}'"
        ),
    )
    args = parser.parse_args()

    config = AgentConfig(
        multi_metrics=args.multi_metrics,
        metric=args.metric,
        metrics_weights=args.metrics_weights_json or AgentConfig().metrics_weights,
    )
    agent = RecClawAgent(config)
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
