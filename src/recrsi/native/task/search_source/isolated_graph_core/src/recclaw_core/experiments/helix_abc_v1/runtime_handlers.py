"""Package-owned, non-training handlers for M1 interface validation.

The handler receives closed typed data only. It does not open files, inspect
environment variables, access a dataset, start a training backend, or expose a
network/import hook.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("zero vector is not a valid smoke input")
    return tuple(value / norm for value in vector)


def _graph_step(
    embeddings: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    # Fixed two-user/two-item bipartite graph. This validates propagation shape
    # and symmetric aggregation without loading data or taking an optimizer step.
    neighbors = ((2, 3), (2,), (0, 1), (0,))
    result: list[tuple[float, ...]] = []
    for node, adjacent in enumerate(neighbors):
        degree = len(adjacent)
        values: list[float] = []
        for column in range(len(embeddings[node])):
            total = 0.0
            for other in adjacent:
                total += embeddings[other][column] / math.sqrt(
                    degree * len(neighbors[other])
                )
            values.append(total)
        result.append(tuple(values))
    return tuple(result)


def _contrastive_loss(left: Sequence[float], right: Sequence[float]) -> float:
    similarity = _dot(_normalize(left), _normalize(right))
    return -math.log(max(1e-9, (1.0 + similarity) / 2.0))


def run_non_training_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the declared mechanism axes on fixed synthetic tensors."""

    required = {
        "candidate_id",
        "mechanism_program_digest",
        "mechanism_semantics_digest",
        "operators",
        "primitives",
        "template_id",
    }
    if set(config) != required:
        raise ValueError("handler config is not the exact closed M1 shape")
    primitives = tuple(str(item) for item in config["primitives"])
    operators = tuple(str(item) for item in config["operators"])
    template_id = str(config["template_id"])

    embeddings = (
        (0.10, 0.20, 0.30),
        (0.30, 0.10, 0.20),
        (0.20, 0.40, 0.10),
        (0.40, 0.20, 0.30),
    )
    exercised: list[str] = ["PAIRWISE_RANKING"]
    user, positive, negative = embeddings[0], embeddings[2], embeddings[3]
    pairwise_margin = _dot(user, positive) - _dot(user, negative)
    interface_loss = math.log1p(math.exp(-pairwise_margin))
    checks: dict[str, Any] = {
        "input_shape": [4, 3],
        "pairwise_margin_finite": math.isfinite(pairwise_margin),
    }

    if "encoder.explicit_message_passing" in primitives:
        propagated = _graph_step(embeddings)
        checks["propagated_shape"] = [len(propagated), len(propagated[0])]
        checks["propagation_finite"] = all(
            math.isfinite(value) for row in propagated for value in row
        )
        exercised.append("GRAPH_PROPAGATION_AGGREGATION")
        if "message.linear_transform" in primitives:
            transformed = tuple(value * 0.5 for value in propagated[0])
            interaction = tuple(
                a * b for a, b in zip(propagated[0], embeddings[0], strict=True)
            )
            checks["operator_branch_finite"] = all(
                math.isfinite(value) for value in transformed + interaction
            )
            exercised.append("PACKAGE_ARCHITECTURE_OPERATOR")

    if "sampler.sampled_unobserved" in primitives:
        sampled_index = int(config["mechanism_semantics_digest"][:2], 16) % 2
        checks["sampled_unobserved_index"] = sampled_index
        exercised.append("NON_DEFAULT_NEGATIVE_SAMPLING")

    if "ssl.objective.info_nce" in primitives:
        ssl_loss = _contrastive_loss(embeddings[0], embeddings[1])
        checks["contrastive_loss_finite"] = math.isfinite(ssl_loss)
        interface_loss += ssl_loss
        exercised.append("SELF_SUPERVISION_CONTRASTIVE")

    if "regularizer.alignment" in primitives or "regularizer.uniformity" in primitives:
        alignment = sum(
            (a - b) ** 2 for a, b in zip(embeddings[0], embeddings[2], strict=True)
        )
        uniformity = math.log(
            math.exp(-sum((a - b) ** 2 for a, b in zip(embeddings[0], embeddings[1], strict=True)))
            + 1e-9
        )
        checks["geometry_terms_finite"] = math.isfinite(alignment + uniformity)
        interface_loss += alignment + abs(uniformity)
        exercised.append("REGULARIZATION_GEOMETRY")

    if template_id == "CONSTRAINT_RANKER_V1":
        constraint = abs(_dot(embeddings[0], embeddings[2])) + abs(
            _dot(embeddings[2], embeddings[3])
        )
        checks["constraint_template_finite"] = math.isfinite(constraint)
        checks["architecture_operators"] = list(operators)
        interface_loss += constraint
        exercised.append("PACKAGE_ARCHITECTURE_OPERATOR")

    signature = hashlib.sha256(
        (
            str(config["candidate_id"])
            + str(config["mechanism_program_digest"])
            + str(config["mechanism_semantics_digest"])
            + template_id
        ).encode("utf-8")
    ).hexdigest()
    if not math.isfinite(interface_loss):
        raise ValueError("non-finite interface loss")
    return {
        "checks": checks,
        "interface_loss": interface_loss,
        "mechanism_axes_exercised": sorted(set(exercised)),
        "optimizer_steps": 0,
        "output_signature": signature,
        "training_backend_started": False,
    }


__all__ = ["run_non_training_smoke"]
