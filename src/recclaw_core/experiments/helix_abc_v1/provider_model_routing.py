"""Station model-ID routing for the shared RecClaw research substrate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


STRONG_MODEL = "gpt-5.6-terra"
EFFICIENT_MODEL = "gpt-5.6-luna"
STRONG_RETURNED_MODEL = STRONG_MODEL
STRONG_RETURNED_MODELS = frozenset({STRONG_RETURNED_MODEL})
EFFICIENT_RETURNED_MODEL = EFFICIENT_MODEL
EFFICIENT_RETURNED_MODELS = frozenset({EFFICIENT_RETURNED_MODEL})
# Luna handles the three focused parent-relative research roles and mechanical
# critique/debugging; Terra owns frontier synthesis and code generation or
# revision.  Start both tiers at medium so model quality is comparable before
# raising both to high only if the real P1 chain exposes a capability deficit.
EFFICIENT_REASONING_EFFORT = "medium"
CONSTRAINED_STRONG_REASONING_EFFORT = "medium"
STRONG_REASONING_EFFORT = "medium"
PRICING_VERSION = "a42d372ccf0b5dd13ecf71203521f9d2"
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-5.4-mini": {
        "cached_input": 0.0975,
        "input": 0.975,
        "output": 5.85,
    },
    "gpt-5.4": {
        "cached_input": 0.325,
        "input": 3.25,
        "output": 19.50,
    },
}

_STRONG_ROLES = frozenset(
    {
        "frontier_architect",
        "implementer",
        "implementation_revision",
        "meta_strategy_synthesis",
        "mechanism_semantic_repair",
        "original_matched_proposal",
        "unspecified_strong",
    }
)
_EFFICIENT_ROLES = frozenset(
    {
        "critic",
        "debugger",
        "falsification_designer",
        "lineage_refiner",
        "mechanism_composer",
    }
)
_CONSTRAINED_STRONG_ROLES = frozenset(
    {
        "frontier_architect",
        "implementer",
        "implementation_revision",
        "mechanism_semantic_repair",
        "meta_strategy_synthesis",
    }
)
_ROLE_REASONING_EFFORT = {
    **{role: EFFICIENT_REASONING_EFFORT for role in _EFFICIENT_ROLES},
    **{
        role: CONSTRAINED_STRONG_REASONING_EFFORT
        for role in _CONSTRAINED_STRONG_ROLES
    },
    **{
        role: STRONG_REASONING_EFFORT
        for role in _STRONG_ROLES - _CONSTRAINED_STRONG_ROLES
    },
}
_ROUTING_PREIMAGE = {
    "efficient_model": EFFICIENT_MODEL,
    "efficient_reasoning_effort": EFFICIENT_REASONING_EFFORT,
    "efficient_roles": sorted(_EFFICIENT_ROLES),
    "schema": "recclaw.provider-model-routing.v1",
    "strong_model": STRONG_MODEL,
    "strong_reasoning_effort": STRONG_REASONING_EFFORT,
    "strong_roles": sorted(_STRONG_ROLES),
    "role_reasoning_effort": {
        role: _ROLE_REASONING_EFFORT[role]
        for role in sorted(_ROLE_REASONING_EFFORT)
    },
}
ROLE_ROUTING_DIGEST = hashlib.sha256(
    json.dumps(
        _ROUTING_PREIMAGE,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderModelSelection:
    role: str
    requested_model: str
    tier: str
    reasoning_effort: str | None
    role_routing_digest: str = ROLE_ROUTING_DIGEST


def select_provider_model(role: str) -> ProviderModelSelection:
    """Return the station model ID assigned to a formal Provider role."""

    normalized = str(role).strip()
    if normalized in _STRONG_ROLES:
        return ProviderModelSelection(
            normalized,
            STRONG_MODEL,
            "STRONG",
            _ROLE_REASONING_EFFORT[normalized],
        )
    if normalized in _EFFICIENT_ROLES:
        return ProviderModelSelection(
            normalized,
            EFFICIENT_MODEL,
            "EFFICIENT",
            EFFICIENT_REASONING_EFFORT,
        )
    raise ValueError(f"unknown Provider model role: {normalized!r}")


def model_routing_manifest() -> dict[str, Any]:
    """Return the pricing-free routing identity persisted by formal manifests."""

    role_models = {
        role: select_provider_model(role).requested_model
        for role in sorted(_STRONG_ROLES | _EFFICIENT_ROLES)
        if role != "unspecified_strong"
    }
    return {
        "efficient_model": EFFICIENT_MODEL,
        "efficient_reasoning_effort": EFFICIENT_REASONING_EFFORT,
        "role_models": role_models,
        "role_reasoning_effort": {
            role: _ROLE_REASONING_EFFORT[role]
            for role in sorted(role_models)
        },
        "role_routing_digest": ROLE_ROUTING_DIGEST,
        "strong_model": STRONG_MODEL,
        "strong_reasoning_effort": STRONG_REASONING_EFFORT,
    }


def summarize_provider_usage(
    traces: Sequence[Mapping[str, Any]], *, round_index: int
) -> dict[str, Any]:
    """Aggregate persisted trace usage without influencing call admission."""

    roles: dict[str, dict[str, int]] = {}
    for trace in traces:
        receipt = trace.get("receipt")
        usage = trace.get("usage")
        if not isinstance(receipt, Mapping) or not isinstance(usage, Mapping):
            continue
        role = str(receipt.get("provider_role") or "unknown")
        row = roles.setdefault(
            role,
            {
                "cached_input_tokens": 0,
                "cache_hit_calls": 0,
                "call_count": 0,
                "implementation_revision_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "semantic_repair_calls": 0,
            },
        )
        cached = int(usage.get("cached_input_tokens") or 0)
        row["call_count"] += 1
        row["input_tokens"] += int(usage.get("input_tokens") or 0)
        row["cached_input_tokens"] += cached
        row["output_tokens"] += int(usage.get("output_tokens") or 0)
        row["cache_hit_calls"] += int(cached > 0)
        row["semantic_repair_calls"] += int(
            role == "mechanism_semantic_repair"
        )
        row["implementation_revision_calls"] += int(
            role == "implementation_revision"
        )
    totals = {
        key: sum(row[key] for row in roles.values())
        for key in next(iter(roles.values()), {
            "cached_input_tokens": 0,
            "cache_hit_calls": 0,
            "call_count": 0,
            "implementation_revision_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "semantic_repair_calls": 0,
        })
    }
    return {
        "pricing_version": PRICING_VERSION,
        "round_index": int(round_index),
        "roles": {role: roles[role] for role in sorted(roles)},
        "schema": "recclaw.provider-usage-summary.v1",
        "totals": totals,
    }


def returned_model_is_compatible(requested_model: str, returned_model: str) -> bool:
    """Require the observed endpoint snapshot for each requested alias."""

    allowed = {
        STRONG_MODEL: STRONG_RETURNED_MODELS,
        EFFICIENT_MODEL: EFFICIENT_RETURNED_MODELS,
    }
    return returned_model in allowed.get(requested_model, frozenset())


def require_returned_model_identity(
    requested_model: str,
    returned_model: str,
) -> None:
    """Reject endpoint alias drift at the shared formal transport boundary."""

    if not returned_model_is_compatible(requested_model, returned_model):
        raise ValueError(
            "Provider returned_model differs from the required endpoint snapshot"
        )


def estimate_provider_usage_cost(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Price persisted usage without affecting Provider call admission."""

    by_model: dict[str, dict[str, Any]] = {}
    for trace in traces:
        receipt = trace.get("receipt")
        usage = trace.get("usage")
        if not isinstance(receipt, Mapping) or not isinstance(usage, Mapping):
            continue
        requested_model = receipt.get("requested_model")
        if not isinstance(requested_model, str):
            role = str(receipt.get("provider_role") or "")
            try:
                requested_model = select_provider_model(role).requested_model
            except ValueError:
                continue
        prices = MODEL_PRICING_USD_PER_MILLION.get(requested_model)
        if prices is None:
            continue
        input_tokens = int(usage.get("input_tokens") or 0)
        cached_tokens = int(usage.get("cached_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        uncached_tokens = max(0, input_tokens - cached_tokens)
        cost = (
            uncached_tokens * prices["input"]
            + cached_tokens * prices["cached_input"]
            + output_tokens * prices["output"]
        ) / 1_000_000
        row = by_model.setdefault(
            requested_model,
            {
                "cached_input_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        row["input_tokens"] += input_tokens
        row["cached_input_tokens"] += cached_tokens
        row["output_tokens"] += output_tokens
        row["estimated_cost_usd"] += cost
    return {
        "models": {model: by_model[model] for model in sorted(by_model)},
        "pricing_version": PRICING_VERSION,
        "schema": "recclaw.provider-usage-cost-audit.v1",
        "total_estimated_cost_usd": sum(
            row["estimated_cost_usd"] for row in by_model.values()
        ),
    }


__all__ = [
    "EFFICIENT_MODEL",
    "EFFICIENT_REASONING_EFFORT",
    "CONSTRAINED_STRONG_REASONING_EFFORT",
    "EFFICIENT_RETURNED_MODEL",
    "EFFICIENT_RETURNED_MODELS",
    "MODEL_PRICING_USD_PER_MILLION",
    "PRICING_VERSION",
    "ProviderModelSelection",
    "ROLE_ROUTING_DIGEST",
    "STRONG_MODEL",
    "STRONG_REASONING_EFFORT",
    "STRONG_RETURNED_MODEL",
    "STRONG_RETURNED_MODELS",
    "estimate_provider_usage_cost",
    "select_provider_model",
    "returned_model_is_compatible",
    "require_returned_model_identity",
    "model_routing_manifest",
    "summarize_provider_usage",
]
