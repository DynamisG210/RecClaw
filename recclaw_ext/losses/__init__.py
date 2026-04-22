"""Local loss stubs for RecClaw method candidates."""

from .alignment import LayerAlignmentLoss
from .bpr_margin import BPRMarginLoss
from .bpr_long_tail import BPRLongTailReweightLoss
from .bpr_norm import BPRNormConstrainedLoss
from .bpr_popreg import BPRPopularityRegularizedLoss
from .rank_aware import RankAwarePairwiseLoss

__all__ = [
    "BPRMarginLoss",
    "BPRLongTailReweightLoss",
    "BPRNormConstrainedLoss",
    "BPRPopularityRegularizedLoss",
    "LayerAlignmentLoss",
    "RankAwarePairwiseLoss",
]
