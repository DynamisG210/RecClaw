from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MutationSpec:
    """One candidate code mutation produced by the GRPO group sampler."""

    dimension: str
    """Action dimension key, e.g. 'tune_lr', 'add_layer'."""

    delta: float
    """Signed step magnitude (positive = increase, negative = decrease)."""

    estimated_cost_secs: float
    """Wall-clock cost estimate used for budget gating."""

    code_diff: str
    """Unified diff string that the trial subagent should apply."""

    code_edits: list[dict] = field(default_factory=list)
    """Structured line-level edits: [{"find": "old line(s)", "replace": "new line(s)"}].
    When non-empty, takes priority over code_diff. Supports multi-line find/replace.
    Use for changes up to ~30 lines. Use code_content for full architecture rewrites."""

    code_content: str = ""
    """Full replacement content for the training script. Use only for architecture-level
    rewrites (>30 lines changed). Overrides both code_edits and code_diff when non-empty."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional proposal metadata preserved for logging / state updates.

    Examples include arm, composite_dims, code_hint, hypothesis, is_jumping,
    predict_code_edits, and predict_code_content. Older versions silently dropped
    these fields, which made downstream ExperimentSkill lessons degrade to
    unknown/unknown.
    """

    def __init__(self, dimension: str, delta: float, estimated_cost_secs: float,
                 code_diff: str, code_edits: list | None = None, code_content: str = "", **extra):
        # Accept and preserve proposal metadata. The trainer only needs the core
        # mutation fields, but the controller needs the metadata for arm rewards,
        # ExperimentSkill, logging, and safe predict.py trial isolation.
        self.dimension = dimension
        self.delta = float(delta)
        self.estimated_cost_secs = float(estimated_cost_secs)
        self.code_diff = code_diff
        self.code_edits = self._normalize_code_edits(code_edits if code_edits is not None else [])
        self.code_content = code_content
        self.extra = dict(extra)

    @staticmethod
    def _normalize_code_edits(code_edits: list) -> list[dict[str, str]]:
        """Accept canonical dict edits and legacy [find, replace] pairs."""
        normalised: list[dict[str, str]] = []
        for edit in code_edits:
            if isinstance(edit, dict):
                normalised.append({
                    "find": str(edit.get("find", "")),
                    "replace": str(edit.get("replace", "")),
                })
            elif isinstance(edit, (list, tuple)) and len(edit) == 2:
                normalised.append({"find": str(edit[0]), "replace": str(edit[1])})
            else:
                normalised.append({"find": "", "replace": ""})
        return normalised

    def to_dict(self) -> dict[str, Any]:
        """Return the full mutation spec, including preserved metadata."""
        data = {
            "dimension": self.dimension,
            "delta": self.delta,
            "estimated_cost_secs": self.estimated_cost_secs,
            "code_diff": self.code_diff,
            "code_edits": self.code_edits,
            "code_content": self.code_content,
        }
        data.update(self.extra)
        return data


@dataclass
class TrialResult:
    """Structured result returned by a trial subagent after executing one MutationSpec."""

    spec: MutationSpec
    wall_time_secs: float
    timed_out: bool
    oom: bool
    val_score: float
    """Higher is better."""

    convergence_trace: list[float] = field(default_factory=list)
    """Validation metric recorded at each epoch/step."""

    error_message: str | None = None
