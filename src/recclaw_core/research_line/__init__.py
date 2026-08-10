"""Unified production spine for the RecClaw Research Line."""

from .interfaces import (
    BehaviorProjection,
    ProducerOutcome,
    ResearchContext,
    ResearchLineInterfaceError,
)
from .runtime import (
    IdeaAcquisitionResult,
    InnovationLaneResult,
    InnovationRuntimeInputs,
    MetaResearchInputs,
    MetaResearchResult,
    ResearchRoundResult,
    SearchCandidate,
    activate_promoted_meta_strategy,
    activate_staged_innovation,
    bindings_for_context,
    resolver_environment_for_profile,
    run_research_round,
)
from .replay import OfflineProducerReplayV1, OfflineReplayError
from .single_round import (
    SingleRoundComposition,
    compose_single_round,
    execute_single_round,
)

__all__ = [
    "BehaviorProjection",
    "ProducerOutcome",
    "ResearchContext",
    "ResearchLineInterfaceError",
    "IdeaAcquisitionResult",
    "InnovationLaneResult",
    "InnovationRuntimeInputs",
    "MetaResearchInputs",
    "MetaResearchResult",
    "ResearchRoundResult",
    "SearchCandidate",
    "activate_promoted_meta_strategy",
    "activate_staged_innovation",
    "bindings_for_context",
    "resolver_environment_for_profile",
    "run_research_round",
    "OfflineProducerReplayV1",
    "OfflineReplayError",
    "SingleRoundComposition",
    "compose_single_round",
    "execute_single_round",
]
