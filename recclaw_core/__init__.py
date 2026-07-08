"""RecClaw-native Candidate Foundry v0.1.

This package is intentionally small: it replays a controlled, no-GPU,
no-LLM, no-RecBole golden fixture for typed memory ingestion, retrieval,
deterministic candidate ranking, and claim-boundary verification.
"""

from recclaw_core.candidate import validate_candidate_card, validate_candidate_cards
from recclaw_core.memory import ingest_bpr_rankcut_memory, retrieve_memory
from recclaw_core.policy import rank_candidates

__all__ = [
    "ingest_bpr_rankcut_memory",
    "rank_candidates",
    "retrieve_memory",
    "validate_candidate_card",
    "validate_candidate_cards",
]
