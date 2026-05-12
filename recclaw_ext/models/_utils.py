"""Small safe helpers for generated RecClaw local models."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a RecBole Config value without assuming dict.get exists."""
    try:
        value = config[key]
    except Exception:  # noqa: BLE001 - RecBole Config is mapping-like, not a plain dict.
        return default
    return default if value is None else value


def config_float(config: Any, key: str, default: float = 0.0) -> float:
    value = config_get(config, key, default)
    if value is None or str(value).strip() == "":
        value = default
    return float(value)


def margin_bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score - float(margin)).mean()


def soft_l2_norm_penalty(
    *embeddings: torch.Tensor | None | float,
    max_norm: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    values = list(embeddings)
    if values and isinstance(values[-1], (int, float)) and max_norm == 1.0:
        # Backward-compatible guard for generated code that called
        # soft_l2_norm_penalty(emb, self.max_norm) before we required keywords.
        max_norm = float(values.pop())
    unexpected = [type(emb).__name__ for emb in values if emb is not None and not isinstance(emb, torch.Tensor)]
    if unexpected:
        raise TypeError(
            "soft_l2_norm_penalty expects tensor embeddings; pass max_norm=... and weight=... by keyword "
            f"instead of positional {unexpected}"
        )
    valid = [emb for emb in values if isinstance(emb, torch.Tensor) and emb.numel() > 0]
    if not valid or weight <= 0:
        device = valid[0].device if valid else torch.device("cpu")
        return torch.zeros((), device=device)
    penalties = [
        F.relu(emb.norm(p=2, dim=-1) - float(max_norm)).pow(2).mean()
        for emb in valid
    ]
    return float(weight) * torch.stack(penalties).mean()
