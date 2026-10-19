"""Machine-owned RecBole trainer semantics for P5/P6/P7 candidates."""

from __future__ import annotations

from functools import wraps
import importlib
from typing import Any


def strict_early_stopping(
    value: float,
    best: float,
    cur_step: int,
    max_step: int,
    bigger: bool = True,
) -> tuple[float, int, bool, bool]:
    """Treat a tied validation score as non-improvement.

    RecBole 1.1.1 resets patience on equality even though ``cur_step`` is
    documented as evaluations that did not *exceed* the best result.  Keep its
    existing stop boundary (``cur_step > max_step``) and change only the
    equality case that made a flat metric run to the epoch ceiling.
    """

    improved = value > best if bigger else value < best
    if improved:
        return value, 0, False, True
    next_step = cur_step + 1
    return best, next_step, next_step > max_step, False


def bind_strict_early_stopping_trainer_class(trainer_class: type[Any]) -> type[Any]:
    """Apply strict tie handling only while this candidate trainer is fitting."""

    if not isinstance(trainer_class, type):
        raise TypeError("trainer_class must be a class")
    if getattr(trainer_class, "_recclaw_machine_owned_strict_early_stop_v1", False):
        return trainer_class
    implementation_fit = trainer_class.fit

    @wraps(implementation_fit)
    def machine_owned_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        trainer_module = importlib.import_module("recbole.trainer.trainer")
        previous = trainer_module.early_stopping
        trainer_module.early_stopping = strict_early_stopping
        try:
            return implementation_fit(self, *args, **kwargs)
        finally:
            trainer_module.early_stopping = previous

    trainer_class.fit = machine_owned_fit
    trainer_class._recclaw_machine_owned_strict_early_stop_v1 = True
    return trainer_class


__all__ = [
    "bind_strict_early_stopping_trainer_class",
    "strict_early_stopping",
]
