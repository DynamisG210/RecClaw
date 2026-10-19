"""Thin program-to-E1 evaluation adapter; no model calls or new trainer.

The launcher supplies the existing E1Runtime, whose native runner binds the
immutable recipe, package, qualification and protected physical worker.
"""
import hashlib
import json
import math
from pathlib import Path

from gagc.e1_runtime import BudgetExhausted


class ProgramEvaluator:
    def __init__(self, runtime, *, worker_timeout_seconds=1800):
        self.runtime = runtime
        self.worker_timeout_seconds = worker_timeout_seconds

    def evaluate(self, program_path):
        path = Path(program_path)
        source = path.read_bytes().decode('utf-8')
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest == self.runtime.parent_sha:
            # Existing measured root, identical to RH's no-op baseline path.
            outcome = {**self.runtime.baseline, "shared_baseline": True,
                       "wall_time_secs": 0.0}
        else:
            budget = self.runtime.budget
            # MAIN reserves genuine capacity in MainResultConsumer. Physical
            # failed/root/duplicate jobs consume real resources, not extra slots.
            timeout = min(self.worker_timeout_seconds, budget.remaining_gpu_secs())
            if timeout < 1:
                raise BudgetExhausted("E1 execution time budget exhausted")
            # Reuse RH's exact metered physical path, not execute(), which would
            # ask RH's implementer to rewrite the other system's program again.
            outcome = self.runtime._train(path, source, timeout, evaluation_reserved=True)
        valid = outcome.get("status") == "ok"
        metrics = {}
        if valid:
            score = float(outcome["ndcg_at_5"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid trusted BERT NDCG@5")
            metrics = {"combined_score": score, "ndcg_at_5": score}
        else:
            # Only native invalid fitness: deliberately no fabricated NDCG key.
            metrics = {"combined_score": -1.0}
        return {"metrics": metrics, "artifacts": {
            "valid": valid, "native_evidence": outcome,
            "source_sha256": digest,
        }}

    def openevolve_result(self, program_path):
        from openevolve.evaluation_result import EvaluationResult
        from resource_stop import SearchResourceStop
        try:
            result = self.evaluate(program_path)
        except BudgetExhausted as stop:
            raise SearchResourceStop(str(stop)) from stop
        # Official artifact ABI accepts text/bytes, not nested Python objects.
        return EvaluationResult(metrics=result["metrics"], artifacts={
            "native_evidence.json": json.dumps(result["artifacts"], sort_keys=True),
        })
