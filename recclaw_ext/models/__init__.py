"""Local recommendation models used by RecClaw."""

from .lightgcn_lw import LightGCNLW
from .lightgcn_residual import LightGCNResidualMix

__all__ = ["LightGCNLW", "LightGCNResidualMix"]
