"""Local feature and score-adjustment stubs for RecClaw candidates."""

from .rerank_adjustments import CoverageBoostReranker, PopularityPenaltyReranker

__all__ = ["CoverageBoostReranker", "PopularityPenaltyReranker"]
