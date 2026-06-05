"""Local recommendation models used by RecClaw."""

from .bpr_margin import BPRMargin
from .bpr_composed import BPRHardNegativeMargin, BPRPopularityAwareMargin
from .bpr_regularized import (
    BPRLongTailReweight,
    BPRNormConstrained,
    BPRPopularityRegularized,
)
from .bpr_sampling import BPRHardNegative, BPRPopularityAwareNegative
from ._losses import (
    BPRLongTailReweightLoss,
    BPRMarginLoss,
    BPRNormConstrainedLoss,
    BPRPopularityRegularizedLoss,
    LayerAlignmentLoss,
    RankAwarePairwiseLoss,
)
from ._utils import config_float, config_get, margin_bpr_loss, soft_l2_norm_penalty
from ._samplers import (
    DebiasedNegativeSampler,
    MixedNegativeSampler,
    MixedNegativeSamplerConfig,
    PopularityAwareNegativeSampler,
)
from .lightgcn_lw import LightGCNLW
from .lightgcn_edge_dropout_residual import LightGCNEdgeDropoutResidualMix
from .lightgcn_edge_dropout_residual_norm_dualpathblend_repair_076 import (
    LightGCNEdgeDropoutResidualNormDualPathBlend,
)
from .lightgcn_objectives import LightGCNAuxAlignment, LightGCNRankAware
from .lightgcn_residual import LightGCNResidualMix
from .lightgcn_residual_norm import (
    LightGCNEdgeDropoutResidualNorm,
    LightGCNResidualNormConstrained,
)
from .lightgcn_sampling import LightGCNDebiasedNegative

__all__ = [
    "BPRHardNegative",
    "BPRHardNegativeMargin",
    "BPRMargin",
    "BPRLongTailReweight",
    "BPRLongTailReweightLoss",
    "BPRMarginLoss",
    "BPRNormConstrained",
    "BPRNormConstrainedLoss",
    "BPRPopularityAwareNegative",
    "BPRPopularityAwareMargin",
    "BPRPopularityRegularized",
    "BPRPopularityRegularizedLoss",
    "config_float",
    "config_get",
    "DebiasedNegativeSampler",
    "LayerAlignmentLoss",
    "LightGCNAuxAlignment",
    "LightGCNEdgeDropoutResidualMix",
    "LightGCNEdgeDropoutResidualNorm",
    "LightGCNEdgeDropoutResidualNormDualPathBlend",
    "LightGCNDebiasedNegative",
    "LightGCNLW",
    "LightGCNRankAware",
    "LightGCNResidualMix",
    "LightGCNResidualNormConstrained",
    "margin_bpr_loss",
    "MixedNegativeSampler",
    "MixedNegativeSamplerConfig",
    "PopularityAwareNegativeSampler",
    "RankAwarePairwiseLoss",
    "soft_l2_norm_penalty",
]
