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
from .scientific_attribution import (
    DeterministicHelixAdmissionV13,
    FrontierEligibilityV2,
    FusedSearchFeedbackV2,
    GuardEvidenceObservationV1,
    GuardEvidenceSnapshotV1,
    NOT_AVAILABLE,
    PromptFeedbackProjectionV2,
    ResearchTaskQueueV1,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchFeedbackClassV2,
    SearchUtilityEventV2,
    ValidationResultBundleV1,
)

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
    "DeterministicHelixAdmissionV13",
    "FrontierEligibilityV2",
    "FusedSearchFeedbackV2",
    "GuardEvidenceObservationV1",
    "GuardEvidenceSnapshotV1",
    "NOT_AVAILABLE",
    "PromptFeedbackProjectionV2",
    "ResearchTaskQueueV1",
    "ResearchTaskStatusV1",
    "ResearchTaskTypeV1",
    "ResearchTaskV1",
    "SearchFeedbackClassV2",
    "SearchUtilityEventV2",
    "ValidationResultBundleV1",
]
