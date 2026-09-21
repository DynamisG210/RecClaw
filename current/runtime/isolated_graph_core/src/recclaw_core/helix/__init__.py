"""M3 composition boundary between Research Line and Evidence Guard."""

from .baseline_calibration import (
    BaselineCalibrationError,
    development_baseline_plan_v31,
    development_lightgcn_recipe_v31,
    finalize_development_baseline_v31,
)
from .contracts import (
    CandidateEnvelope,
    CompactFeedback,
    GuardContext,
    PortAdjudication,
    RawResultEnvelope,
)
from .fusion import DeterministicHelixFusionV1, HelixFusionBridgeV1
from .frontier_allocation import (
    AllocationActionV31,
    AllocationClosureV31,
    FrontierEvidenceAllocatorV31,
    HelixAllocationDecisionV31,
    HelixAllocationPolicyV31,
)
from .guard_adapter import EvidenceGuardPortV1
from .ledger import EvidenceGuardLedgerWriterV1
from .matched_budget import (
    MatchedBudgetAuditV31,
    audit_matched_budget_contract_v31,
    require_matched_budget_contract_v31,
)
from .outer_panel import (
    OuterPanelError,
    seal_outer_panel_v31,
    validate_outer_panel_manifest_v31,
)
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
    ValueOfInformationHelixAdmissionV30,
)

__all__ = [
    "CandidateEnvelope",
    "BaselineCalibrationError",
    "AllocationActionV31",
    "AllocationClosureV31",
    "CompactFeedback",
    "DeterministicHelixFusionV1",
    "EvidenceGuardLedgerWriterV1",
    "EvidenceGuardPortV1",
    "FrontierEvidenceAllocatorV31",
    "GuardContext",
    "HelixFusionBridgeV1",
    "HelixAllocationDecisionV31",
    "HelixAllocationPolicyV31",
    "MatchedBudgetAuditV31",
    "NullEvidencePortV1",
    "OuterPanelError",
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
    "ValueOfInformationHelixAdmissionV30",
    "audit_matched_budget_contract_v31",
    "development_baseline_plan_v31",
    "development_lightgcn_recipe_v31",
    "finalize_development_baseline_v31",
    "require_matched_budget_contract_v31",
    "seal_outer_panel_v31",
    "validate_outer_panel_manifest_v31",
]
