from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gagc.state import ThompsonState

from gagc.schemas import MutationSpec


# ---------------------------------------------------------------------------
# Arm coupling definitions (Amazon Reviews benchmark)
# ---------------------------------------------------------------------------

# Composite arms: synergistically coupled dims treated as one arm.
# The arm name maps to the list of original dimensions it represents.
COMPOSITE_ARMS: dict[str, list[str]] = {
    "tune_lr_batch": ["tune_lr", "tune_batch_size"],
    "tune_dropout_wd": ["tune_dropout", "tune_weight_decay"],
    "tune_lr_batch_scheduler": ["tune_lr", "tune_batch_size", "add_lr_scheduler"],
}

# Interference coupling: arms in the same group are mutually exclusive per round.
MUTEX_GROUPS: list[set[str]] = [
    {"change_architecture", "change_loss_function", "change_optimizer_type"},
    {"change_architecture", "tune_dropout", "tune_weight_decay", "add_regularization"},
    {"change_architecture", "tune_activation", "tune_normalization"},
]

# Basin-jumping arms: structural changes that shift the optimisation basin.
JUMPING_DIMS: set[str] = {"change_architecture", "change_loss_function", "change_optimizer_type"}


# ---------------------------------------------------------------------------
# Arm coupling definitions (KuaiRec / GR benchmark)
# ---------------------------------------------------------------------------

GR_COMPOSITE_ARMS: dict[str, list[str]] = {
    "tune_optimizer_schedule": ["tune_lr", "tune_batch_size", "add_lr_scheduler"],
    "tune_loss_balance": ["tune_cls_weight", "tune_huber_weight"],
    "tune_vocab_quantization": ["tune_q_start", "tune_q_end", "tune_q_decay"],
    "tune_transformer_capacity": ["tune_hidden_dim", "tune_num_heads", "tune_dec_layers"],
}

GR_MUTEX_GROUPS: list[set[str]] = [
    {"change_decoder_backbone", "toggle_embedding_mixup"},
]

GR_JUMPING_DIMS: set[str] = {"change_decoder_backbone"}


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------

def thompson_sample(
    state: "ThompsonState",
    candidate_arms: list[str],
    mutex_groups: list[set[str]],
    G: int = 4,
) -> list[str]:
    """Sample G arms via Thompson Sampling with mutual-exclusion filtering.

    Each arm draws one sample from Beta(alpha, beta). The top-2G arms by
    sampled value are passed to filter_selection() which enforces mutex groups
    and returns at most G arms.
    """
    scores: dict[str, float] = {}
    for arm in candidate_arms:
        arm_state = state.arms.get(arm)
        if arm_state is None:
            alpha, beta = 1.0, 1.0
        else:
            alpha, beta = arm_state.alpha, arm_state.beta
        scores[arm] = random.betavariate(alpha, beta)

    top_k = sorted(candidate_arms, key=lambda a: scores[a], reverse=True)[: 2 * G]
    return filter_selection(top_k, mutex_groups, G)


def filter_selection(
    ranked_arms: list[str],
    mutex_groups: list[set[str]],
    G: int,
) -> list[str]:
    """Greedily select up to G arms, blocking mutex partners of already-selected arms."""
    selected: list[str] = []
    seen: set[str] = set()
    blocked: set[str] = set()
    for arm in ranked_arms:
        if arm in seen:
            continue
        if arm in blocked:
            continue
        selected.append(arm)
        seen.add(arm)
        for group in mutex_groups:
            if arm in group:
                blocked |= group
        if len(selected) == G:
            break
    return selected


# ---------------------------------------------------------------------------
# GRPO advantage normalisation (unchanged)
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standard GRPO normalisation: Â_i = (r_i - μ) / (σ + ε).

    Returns a list of the same length as rewards. When all rewards are identical
    (σ ≈ 0) advantages are all zero — no gradient signal, which is correct.
    """
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    variance = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(variance)
    return [(r - mu) / (sigma + eps) for r in rewards]
