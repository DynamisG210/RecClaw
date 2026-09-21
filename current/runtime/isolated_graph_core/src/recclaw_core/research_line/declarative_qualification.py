"""Family-neutral behavioral check used after the shared mechanical qualifier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


def declarative_behavioral_unit_check(
    *, execution_contract: Mapping[str, Any]
) -> Callable[[Any, Any, Any], None]:
    expected = str(execution_contract["config"]["recclaw_model_type"])

    def check(model: Any, config: Any, dataset: Any) -> None:
        from recbole.model.abstract_recommender import (
            GeneralRecommender,
            SequentialRecommender,
        )

        expected_class = (
            SequentialRecommender
            if expected == "SEQUENTIAL"
            else GeneralRecommender
        )
        if not isinstance(model, expected_class):
            raise AssertionError(
                f"candidate does not implement declared {expected} RecBole family"
            )
        for method_name in ("calculate_loss", "predict", "full_sort_predict"):
            if method_name not in model.__class__.__dict__:
                raise AssertionError(
                    f"candidate must directly implement {method_name}"
                )
        primitive_ids = tuple(
            getattr(model.__class__, "__recclaw_implemented_primitive_ids__", ())
        )
        component_ids = tuple(
            getattr(model.__class__, "__recclaw_implemented_component_ids__", ())
        )
        component_specs = getattr(
            model.__class__, "__recclaw_implemented_component_specs__", None
        )
        if not primitive_ids and not component_ids:
            raise AssertionError("candidate exposes no compiled mechanism identity")
        if not isinstance(component_specs, Mapping) or set(component_specs) != set(
            component_ids
        ):
            raise AssertionError(
                "candidate component metadata differs from compiled implementation"
            )
        if getattr(model.__class__, "__recclaw_static_candidate__", False):
            raise AssertionError("static candidate selection is forbidden")
        if getattr(model.__class__, "__recclaw_hidden_fallback__", False):
            raise AssertionError("hidden fallback is forbidden")

    return check


__all__ = ["declarative_behavioral_unit_check"]


