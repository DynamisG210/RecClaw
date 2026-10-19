from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


_DEFAULT_EXPERIMENT_SKILL = """\
# RecHarness Experiment Skill

## Objective
Use benchmark val_score as the search and winner criterion.
Do not use held-out test metrics during search.
Thompson Sampling chooses high-level strategy arms; this skill only guides how to instantiate
small runnable code edits within the selected strategy. Strategy arms are proposal policies,
not scalar knobs. One freeform discovery strategy may appear each search round to expose
mechanisms missing from the fixed strategy set.

## Recent Patch Lessons
No learned patch lessons yet.

## Concrete Avoids
No concrete patch-level avoids yet.
"""


@dataclass
class ArmState:
    name: str
    alpha: float = 1.0   # positive reward accumulator (prior = 1)
    beta: float = 1.0    # negative reward accumulator (prior = 1)


@dataclass
class ThompsonState:
    arms: dict[str, ArmState] = field(default_factory=dict)
    current_basin_id: str = "default"
    basin_arms: dict[str, dict[str, ArmState]] = field(default_factory=dict)
    global_budget: float = 3600.0
    # SkillOpt-inspired memory
    score_history: list[float] = field(default_factory=list)    # per-round best validation score
    rejected_dims_buffer: list[dict] = field(default_factory=list)  # [{dim, round_idx, reason}]
    skill_notes: str = "Cold start: no prior knowledge."        # strategy text summary for LLM
    # Basin-jumping support
    pending_queue: list[dict] = field(default_factory=list)     # [{dim, t_start}]
    provisional_incumbent: dict = field(default_factory=dict)    # best code snapshot before provisional jump
    round_idx: int = 0
    baseline_done: bool = False
    # ExperimentSkill patch-level memory
    recent_text_gradients: list[str] = field(default_factory=list)  # last N winner lessons (ring buffer, max 3)
    failure_memory: list[dict] = field(default_factory=list)         # concrete avoid rules [{round_idx, arm, failure_type, patch_pattern, avoid_rule, count}]
    experiment_memory: list[dict] = field(default_factory=list)      # raw experiment digests [{round_idx, arm, result_summary, ...}]
    experiment_skill: str = _DEFAULT_EXPERIMENT_SKILL
    processed_trial_group_ids: list[str] = field(default_factory=list) # idempotency guard for update_thompson_state

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        data = {
            "arms": {k: {"name": v.name, "alpha": v.alpha, "beta": v.beta}
                     for k, v in self.arms.items()},
            "current_basin_id": self.current_basin_id,
            "basin_arms": {
                basin_id: {
                    k: {"name": v.name, "alpha": v.alpha, "beta": v.beta}
                    for k, v in arms.items()
                }
                for basin_id, arms in self.basin_arms.items()
            },
            "global_budget": self.global_budget,
            "score_history": self.score_history,
            "rejected_dims_buffer": self.rejected_dims_buffer,
            "skill_notes": self.skill_notes,
            "pending_queue": self.pending_queue,
            "provisional_incumbent": self.provisional_incumbent,
            "round_idx": self.round_idx,
            "baseline_done": self.baseline_done,
            "recent_text_gradients": self.recent_text_gradients,
            "failure_memory": self.failure_memory,
            "experiment_memory": self.experiment_memory,
            "experiment_skill": self.experiment_skill,
            "processed_trial_group_ids": self.processed_trial_group_ids,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ThompsonState":
        data = json.loads(text)

        # Backward-compatible: old format had "dimensions" — ignore, start fresh
        # (arms will be populated by the caller with _active_arms())
        arms: dict[str, ArmState] = {}
        if "arms" in data:
            for k, v in data["arms"].items():
                arms[k] = ArmState(
                    name=v.get("name", k),
                    alpha=float(v.get("alpha", 1.0)),
                    beta=float(v.get("beta", 1.0)),
                )

        basin_arms: dict[str, dict[str, ArmState]] = {}
        if "basin_arms" in data:
            for basin_id, raw_arms in data.get("basin_arms", {}).items():
                basin_arms[str(basin_id)] = {
                    k: ArmState(
                        name=v.get("name", k),
                        alpha=float(v.get("alpha", 1.0)),
                        beta=float(v.get("beta", 1.0)),
                    )
                    for k, v in raw_arms.items()
                }

        state = cls(
            arms=arms,
            current_basin_id=str(data.get("current_basin_id", "default")),
            basin_arms=basin_arms,
            global_budget=float(data.get("global_budget", 3600.0)),
            score_history=list(data.get("score_history", [])),
            rejected_dims_buffer=list(data.get("rejected_dims_buffer", [])),
            skill_notes=str(data.get("skill_notes", "Cold start: no prior knowledge.")),
            pending_queue=list(data.get("pending_queue", [])),
            provisional_incumbent=dict(data.get("provisional_incumbent", {})),
            round_idx=int(data.get("round_idx", 0)),
            baseline_done=bool(data.get("baseline_done", bool(data.get("score_history")))),
            recent_text_gradients=list(data.get("recent_text_gradients", [])),
            failure_memory=list(data.get("failure_memory", [])),
            experiment_memory=list(data.get("experiment_memory", [])),
            experiment_skill=str(data.get("experiment_skill", _DEFAULT_EXPERIMENT_SKILL)),
            processed_trial_group_ids=list(data.get("processed_trial_group_ids", [])),
        )
        return state

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def get_or_create_arm(self, name: str) -> ArmState:
        if name not in self.arms:
            self.arms[name] = ArmState(name=name)
        return self.arms[name]

    def success_rate(self, arm_name: str) -> float:
        """Expected value of Beta(alpha, beta) = alpha / (alpha + beta)."""
        arm = self.arms.get(arm_name)
        if arm is None:
            return 0.5
        return arm.alpha / (arm.alpha + arm.beta)
