"""M3 composition boundary between Research Line and Evidence Guard."""

from .contracts import (
    CandidateEnvelope,
    CompactFeedback,
    GuardContext,
    PortAdjudication,
    RawResultEnvelope,
)
from .fusion import DeterministicHelixFusionV1, HelixFusionBridgeV1
from .guard_adapter import EvidenceGuardPortV1
from .ledger import EvidenceGuardLedgerWriterV1
from .ports import NullEvidencePortV1

__all__ = [
    "CandidateEnvelope",
    "CompactFeedback",
    "DeterministicHelixFusionV1",
    "EvidenceGuardLedgerWriterV1",
    "EvidenceGuardPortV1",
    "GuardContext",
    "HelixFusionBridgeV1",
    "NullEvidencePortV1",
    "PortAdjudication",
    "RawResultEnvelope",
]
