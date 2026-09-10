"""P4 Sparse-Spectral search-space adapter.

This adapter binds Research Line orchestration to the P4 mechanism family
without routing it through the BL-ICF compiler or the mixed expanded_v2 pool.
The frozen strong comparator is FaGSP; sparse-linear anchors whose executable
identity was not present in the handoff are retained only as parent metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    content_id,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    P4_SPARSE_SPECTRAL_EVALUATOR,
    P4_SPARSE_SPECTRAL_SPLIT,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    OpenSpecProjectionError,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchCandidateBindingV1,
    SearchExecutableEntryV1,
    SearchExecutableProfileV1,
    SearchProfileActivationV1,
    SearchProfileEntryOriginV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    OpenResearchSpecV1,
)
from recclaw_core.research_line.interfaces import ProducerOutcome

from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceExecutionBindingV1,
)


P4_PROFILE_ID = "p4_sparse_spectral_v1"
P4_ADAPTER_ID = "recclaw.search-space-adapter.p4-sparse-spectral.v1"
P4_PROTOCOL_REF = "recclaw.p4-sparse-spectral.ml1m-rs811-fullsort.v1"
P4_RAW_ML1M_SHA256 = (
    "e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389"
)
P4_SEARCH_INCUMBENT_REF = "p4-baseline:FaGSP:seed2026-validation"
P4_SEARCH_INCUMBENT_NDCG_AT_10 = 0.24550679209020307
P4_FINAL_COMPARATOR_REF = "p4-baseline:FaGSP:fixed-ml1m-test-panel"
P4_FINAL_COMPARATOR_TEST_NDCG_AT_10_MEAN = 0.315787
P4_EXECUTABLE_PARENT_ID = "p4-parent:fagsp:fixed-ml1m"
P4_FAGSP_RUNNER_RELATIVE_PATH = (
    "scripts/p4_baselines/run_fagsp_recbole_protocol.py"
)

# Compatibility names used by the standalone CLI. "Primary" means the
# online search frontier, which must be validation-only.
P4_PRIMARY_COMPARATOR_REF = P4_SEARCH_INCUMBENT_REF
P4_PRIMARY_COMPARATOR_NDCG_AT_10 = P4_SEARCH_INCUMBENT_NDCG_AT_10


P4_FAGSP_SEARCH_INCUMBENT_IDENTITY = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.fagsp-search-incumbent.v1",
        "comparator_ref": P4_SEARCH_INCUMBENT_REF,
        "runner_sha256": (
            "823380f1ba659696907b2f286cae144f7c2a4f2394746b9552863cd8c0ca5cf9"
        ),
        "qualification_result_sha256": (
            "d1382a146aa2e92bf05760d489ad3fdaab736eb11c071024b94925f745661007"
        ),
        "seed": 2026,
        "partition_role": "P4_VALIDATION_SELECTION",
        "metric_source": "FROZEN_RECBole_RS811_VALIDATION_RESULT",
        "validation_ndcg_at_10": P4_SEARCH_INCUMBENT_NDCG_AT_10,
        "test_access_during_search": "FORBIDDEN",
        "split_identity": {
            "train": "307b18468164a419374ee6c6b8e8ae6cf36c9dc3a2eb1c53d1c08e855f12618f",
            "validation": "3e7a621ab35649c6e8bbda653612c945d82eafc44f8b9eba870f8b2a0196cdba",
            "test": "46a0077943aadec63f02c281ff2e1ee7dda96d8c7d03379d9b5faf715af3b451",
        },
        "official_revision": "d97d6d8e37d5ca1dd2c6183b2548c20c4b410d25",
        "protocol_ref": P4_PROTOCOL_REF,
        "raw_dataset_sha256": P4_RAW_ML1M_SHA256,
    }
)
P4_FAGSP_SEARCH_INCUMBENT_DIGEST = sha256_digest(
    P4_FAGSP_SEARCH_INCUMBENT_IDENTITY
)


P4_FAGSP_FINAL_COMPARATOR_IDENTITY = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.fagsp-final-comparator.v1",
        "comparator_ref": P4_FINAL_COMPARATOR_REF,
        "runner_sha256": (
            "823380f1ba659696907b2f286cae144f7c2a4f2394746b9552863cd8c0ca5cf9"
        ),
        "fixed_replication_manifest_sha256": (
            "8a4d5763e0703f139a912062aa36e22b4266b43bb6eabbd0070e4a613794a1e5"
        ),
        "result_sha256_by_seed": {
            "2026": "e051a8574524090811e6d2153bea984dc7ef0ad9acc84db0f05f81bce36dc665",
            "2027": "67ba4ac2c2d012ea02210db8327debe5ff29e344829dd1333a069bb1a3331424",
            "2028": "6e4835866d2dfc8e7fdb167d67070c3bbf53981d933697a31b76122a8d76e9de",
        },
        "test_ndcg_at_10_by_seed": {
            "2026": 0.31476868642729394,
            "2027": 0.31376293177102504,
            "2028": 0.31882943572197187,
        },
        "test_ndcg_at_10_mean": P4_FINAL_COMPARATOR_TEST_NDCG_AT_10_MEAN,
        "test_ndcg_at_10_population_std": 0.002190,
        "partition_role": "FINAL_CONFIRMATION_TEST",
        "admission_rule": "FROZEN_CANDIDATE_ONLY_AFTER_SEARCH",
        "official_revision": "d97d6d8e37d5ca1dd2c6183b2548c20c4b410d25",
        "protocol_ref": P4_PROTOCOL_REF,
        "raw_dataset_sha256": P4_RAW_ML1M_SHA256,
    }
)
P4_FAGSP_FINAL_COMPARATOR_DIGEST = sha256_digest(
    P4_FAGSP_FINAL_COMPARATOR_IDENTITY
)

# Backward-compatible import names now intentionally resolve to the online
# validation identity, never the final test panel.
P4_FAGSP_COMPARATOR_IDENTITY = P4_FAGSP_SEARCH_INCUMBENT_IDENTITY
P4_FAGSP_COMPARATOR_DIGEST = P4_FAGSP_SEARCH_INCUMBENT_DIGEST


P4_FAGSP_PARENT_EXECUTION_IDENTITY = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.fagsp-parent-execution.v1",
        "parent_id": P4_EXECUTABLE_PARENT_ID,
        "runner_relative_path": P4_FAGSP_RUNNER_RELATIVE_PATH,
        "runner_sha256": (
            "823380f1ba659696907b2f286cae144f7c2a4f2394746b9552863cd8c0ca5cf9"
        ),
        "qualification_result_sha256": (
            "d1382a146aa2e92bf05760d489ad3fdaab736eb11c071024b94925f745661007"
        ),
        "search_incumbent_identity_digest": P4_FAGSP_SEARCH_INCUMBENT_DIGEST,
        "official_revision": "d97d6d8e37d5ca1dd2c6183b2548c20c4b410d25",
        "fixed_parameters": {
            "pri_factor1": 256,
            "pri_factor2": 128,
            "alpha1": 0.3,
            "alpha2": 0.5,
            "order1": 12,
            "order2": 14,
            "q": 0.7,
        },
        "operator_contract": (
            "train-only binary user-item interaction matrix",
            "degree-normalized interaction SVD",
            "two exact singular-spectrum frequency-complement cascades",
            "quantile reweighting of observed interactions",
            "hierarchical low-pass reconstruction",
            "full-sort validation with train interactions masked",
        ),
        "candidate_rule": (
            "preserve the frozen FaGSP parent equations except for the exact "
            "mechanism change declared by the OpenSpec"
        ),
    }
)
P4_FAGSP_PARENT_EXECUTION_DIGEST = sha256_digest(
    P4_FAGSP_PARENT_EXECUTION_IDENTITY
)


_P4_PROFILE_ASSET_SHA256 = {
    P4_FAGSP_RUNNER_RELATIVE_PATH: (
        "823380f1ba659696907b2f286cae144f7c2a4f2394746b9552863cd8c0ca5cf9"
    ),
    "configs/p4_baselines/fagsp_seed2026_qualification.json": (
        "d1382a146aa2e92bf05760d489ad3fdaab736eb11c071024b94925f745661007"
    ),
    "configs/p4_baselines/P4_FAGSP_FIXED_REPLICATION_MANIFEST.json": (
        "8a4d5763e0703f139a912062aa36e22b4266b43bb6eabbd0070e4a613794a1e5"
    ),
    "configs/p4_baselines/fagsp_seed2026_full.json": (
        "e051a8574524090811e6d2153bea984dc7ef0ad9acc84db0f05f81bce36dc665"
    ),
    "configs/p4_baselines/fagsp_seed2027.json": (
        "67ba4ac2c2d012ea02210db8327debe5ff29e344829dd1333a069bb1a3331424"
    ),
    "configs/p4_baselines/fagsp_seed2028.json": (
        "6e4835866d2dfc8e7fdb167d67070c3bbf53981d933697a31b76122a8d76e9de"
    ),
}


def validate_p4_profile_assets(repo_root: Path) -> Mapping[str, str]:
    root = Path(repo_root).resolve()
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in _P4_PROFILE_ASSET_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"P4 frozen profile asset is missing: {relative_path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"P4 frozen profile asset identity mismatch: {relative_path}"
            )
        observed[relative_path] = actual_sha256
    return canonical_value(observed)


P4_PROTOCOL = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.protocol.v1",
        "dataset": "ml-1m",
        "raw_dataset_sha256": P4_RAW_ML1M_SHA256,
        "split": {"RS": (0.8, 0.1, 0.1)},
        "group_by": "user",
        "order": "RO",
        "mode": "full",
        "metric": "NDCG@10",
        "topk": 10,
        "fit_partition": "train only",
        "selection_partition": "validation",
        "final_confirmation_test_mask": "train plus validation",
        "search_seed": 2026,
        "final_confirmation_seeds": (2026, 2027, 2028),
        "train_batch_size": 2048,
        "eval_batch_size": 65536,
        "worker": 8,
        "claim_scope": "single-seed search is discovery only",
    }
)
P4_PROTOCOL_DIGEST = sha256_digest(P4_PROTOCOL)


P4_BASELINE_PACK = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.baseline-pack.v1",
        "search_incumbent": {
            "name": "FaGSP",
            "candidate_id": P4_SEARCH_INCUMBENT_REF,
            "validation_ndcg_at_10": P4_SEARCH_INCUMBENT_NDCG_AT_10,
            "seed": 2026,
            "partition_role": "P4_VALIDATION_SELECTION",
            "test_access_during_search": "FORBIDDEN",
            "runner_sha256": (
                "823380f1ba659696907b2f286cae144f7c2a4f2394746b9552863cd8c0ca5cf9"
            ),
            "official_revision": "d97d6d8e37d5ca1dd2c6183b2548c20c4b410d25",
            "identity_status": "RUNNER_AND_RESULT_IDENTITY_AVAILABLE",
            "comparator_identity_digest": P4_FAGSP_SEARCH_INCUMBENT_DIGEST,
        },
        "final_comparator": {
            "name": "FaGSP",
            "candidate_id": P4_FINAL_COMPARATOR_REF,
            "test_ndcg_at_10_mean": P4_FINAL_COMPARATOR_TEST_NDCG_AT_10_MEAN,
            "test_ndcg_at_10_population_std": 0.002190,
            "seeds": (2026, 2027, 2028),
            "partition_role": "FINAL_CONFIRMATION_TEST",
            "admission_rule": "FROZEN_CANDIDATE_ONLY_AFTER_SEARCH",
            "comparator_identity_digest": P4_FAGSP_FINAL_COMPARATOR_DIGEST,
        },
        "strong_pack": (
            {
                "name": "FaGSP",
                "role": "primary_spectral_parent_and_final_comparator",
                "test_ndcg_at_10_mean": 0.315787,
                "identity_status": "RUNNER_AND_RESULT_IDENTITY_AVAILABLE",
            },
            {
                "name": "ADMMSLIM",
                "role": "strong_sparse_linear_parent",
                "test_ndcg_at_10_mean": 0.312600,
                "identity_status": "SOURCE_OR_STATUS_ONLY_NEEDS_FROZEN_CONFIG_RESULT_IDENTITY",
            },
            {
                "name": "ChebyCF",
                "role": "strong_spectral_parent",
                "test_ndcg_at_10_mean": 0.307573,
                "test_ndcg_at_10_population_std": 0.002353,
                "runner_sha256": (
                    "70befefd2809f44ab3ea1e0189af6d324c22fee4df2a91ce1e42568f9ad341ed"
                ),
                "official_revision": "6961667fa45132dfe396df8be45fa49021037a03",
                "identity_status": "RUNNER_AND_RESULT_IDENTITY_AVAILABLE",
            },
            {
                "name": "BSPM",
                "role": "strong_sparse_spectral_parent",
                "test_ndcg_at_10_mean": 0.302705,
                "test_ndcg_at_10_population_std": 0.000947,
                "identity_status": "RUNNER_AND_RESULT_IDENTITY_AVAILABLE_FROM_READABLE_EVIDENCE",
            },
        ),
        "additional_parents": (
            {
                "name": "EASE",
                "role": "sparse_linear_parent",
                "test_ndcg_at_10_mean": 0.302367,
                "identity_status": "SOURCE_OR_STATUS_ONLY_NEEDS_FROZEN_CONFIG_RESULT_IDENTITY",
            },
            {
                "name": "SGFCF",
                "role": "gsp_mechanism_representative",
                "test_ndcg_at_10_mean": 0.301913,
                "test_ndcg_at_10_population_std": 0.002052,
                "runner_sha256": (
                    "9574ffb7d4963f71fdf7c83028c04478c34490dec34150f5fa957cc25c290a3b"
                ),
                "official_revision": "e5736216a03afc165d714c7e78763f4123299942",
                "identity_status": "RUNNER_AND_RESULT_IDENTITY_AVAILABLE",
            },
            {
                "name": "SLIMElastic",
                "role": "sparse_linear_parent",
                "test_ndcg_at_10_mean": 0.297967,
                "identity_status": "SOURCE_OR_STATUS_ONLY_NEEDS_FROZEN_CONFIG_RESULT_IDENTITY",
            },
        ),
    }
)


P4_PARENT_REGISTRY = canonical_value(
    {
        "schema": "recclaw.p4-sparse-spectral.parent-registry.v1",
        "profile_id": P4_PROFILE_ID,
        "parents": (
            {
                "parent_id": P4_EXECUTABLE_PARENT_ID,
                "mechanism_family": "spectral_graph_signal_filter",
                "evidence_status": "FROZEN_RUNNER_AND_THREE_SEED_RESULTS",
                "research_line_execution_admission": "PROFILE_OWNED_FROZEN_RUNNER",
                "execution_identity_digest": P4_FAGSP_PARENT_EXECUTION_DIGEST,
                "runner_relative_path": P4_FAGSP_RUNNER_RELATIVE_PATH,
            },
            {
                "parent_id": "p4-parent:chebycf:fixed-ml1m",
                "mechanism_family": "chebyshev_spectral_filter",
                "evidence_status": "FROZEN_RUNNER_AND_THREE_SEED_RESULTS",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
            {
                "parent_id": "p4-parent:bspm:fixed-ml1m",
                "mechanism_family": "bipartite_spectral_propagation",
                "evidence_status": "READABLE_RUNNER_AND_THREE_SEED_RESULTS",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
            {
                "parent_id": "p4-parent:sgfcf:fixed-ml1m",
                "mechanism_family": "graph_signal_processing",
                "evidence_status": "FROZEN_RUNNER_AND_THREE_SEED_RESULTS",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
            {
                "parent_id": "p4-parent:admmslim:status-anchor",
                "mechanism_family": "sparse_linear_admm",
                "evidence_status": "STATUS_AND_OLD_PILOT_ONLY",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
            {
                "parent_id": "p4-parent:ease:status-anchor",
                "mechanism_family": "closed_form_sparse_linear",
                "evidence_status": "SOURCE_AND_STATUS_ONLY",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
            {
                "parent_id": "p4-parent:slimelastic:status-anchor",
                "mechanism_family": "elastic_net_sparse_linear",
                "evidence_status": "SOURCE_AND_STATUS_ONLY",
                "research_line_execution_admission": "REFERENCE_ONLY",
            },
        ),
        "activation_rule": (
            "FaGSP is the executable frozen v1 parent; every other parent "
            "remains reference-only until separately qualified"
        ),
    }
)


P4_PROFILE_MANIFEST = canonical_value(
    {
        "schema": "recclaw.search-space-profile.p4-sparse-spectral.v1",
        "profile_id": P4_PROFILE_ID,
        "adapter_id": P4_ADAPTER_ID,
        "protocol_ref": P4_PROTOCOL_REF,
        "protocol_digest": P4_PROTOCOL_DIGEST,
        "common_prediction_object": "score_u = x_u W or score_u = F(R/A)x_u",
        "taxonomy_gate": {
            "included": (
                "static interaction-operator CF",
                "sparse or regularized coefficient operators",
                "explicit graph-signal, spectral, or low-rank operators",
            ),
            "excluded": (
                "learned embedding GNN/contrastive BL-ICF models",
                "chronological sequence encoders",
                "semantic-ID generative recommenders",
                "diffusion or flow reconstruction models",
            ),
        },
        "baseline_pack": P4_BASELINE_PACK,
        "parent_registry": P4_PARENT_REGISTRY,
        "router_scale": {
            "search_frontier": P4_SEARCH_INCUMBENT_REF,
            "search_frontier_partition_role": "P4_VALIDATION_SELECTION",
            "search_frontier_ndcg_at_10": P4_SEARCH_INCUMBENT_NDCG_AT_10,
            "final_comparator": P4_FINAL_COMPARATOR_REF,
            "final_comparator_partition_role": "FINAL_CONFIRMATION_TEST",
            "raw_cross_profile_ndcg_competition": False,
        },
        "memory_namespace": "SEARCH_MEMORY/P4_SPARSE_SPECTRAL_V1",
        "results_namespace": "p4_sparse_spectral_v1",
    }
)
P4_PROFILE_DIGEST = sha256_digest(P4_PROFILE_MANIFEST)


@dataclass(frozen=True, slots=True)
class P4SparseSpectralCandidateV1:
    spec: OpenResearchSpecV1
    candidate_id: str
    producer_role: str
    mechanism_id: str
    mechanism_axis: str
    mechanism_program: Mapping[str, Any]
    parent_candidate_id: str | None
    utility_features: SearchUtilityFeaturesV1
    feature_evidence: RouterFeatureEvidenceV1
    semantic_identity_ref: str
    semantic_identity_digest: str
    mechanism_program_digest: str
    candidate_package_ref: str
    candidate_package_digest: str
    execution_contract: Mapping[str, Any]
    capability_ref: str
    capability_digest: str
    executable_entrypoint: str
    qualification_receipt_ref: str
    qualification_receipt_digest: str

    schema = "recclaw.p4-sparse-spectral.search-candidate.v1"

    def __post_init__(self) -> None:
        if self.spec.execution_contract is None:
            raise ValueError("P4 candidate requires an execution contract")
        execution_contract = canonical_value(dict(self.execution_contract))
        if execution_contract != canonical_value(
            dict(self.spec.execution_contract)
        ):
            raise ValueError("P4 candidate execution contract drift")
        object.__setattr__(self, "execution_contract", execution_contract)
        object.__setattr__(
            self,
            "mechanism_program",
            canonical_value(dict(self.mechanism_program)),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            method = getattr(value, "to_dict", None)
            if callable(method):
                return method()
            return value

        return canonical_value(
            {
                "schema": self.schema,
                **{
                    field.name: encode(getattr(self, field.name))
                    for field in fields(self)
                },
            }
        )


def p4_frozen_profile_ref() -> Mapping[str, Any]:
    return canonical_value(
        {
            "profile_id": P4_PROFILE_ID,
            "profile_digest": P4_PROFILE_DIGEST,
            "profile_kind": "OFFLINE_TOPN",
        }
    )


def p4_baseline_context(parent_execution_record: Mapping[str, Any]) -> Mapping[str, Any]:
    from .p4_runtime import actual_execution_contract
    contract = actual_execution_contract(parent_execution_record)
    return canonical_value({
        "schema": "recclaw.p4-residual-parent-context.v1",
        "search_space_profile": P4_PROFILE_ID,
        "protocol": P4_PROTOCOL,
        "parent_execution_record": parent_execution_record,
        "parent_anchor": {"name": "FaGSP", "execution_contract": contract,
            "historical_execution_identity": P4_FAGSP_PARENT_EXECUTION_IDENTITY,
            "paired_metric": {"seed": 2026, "value": P4_SEARCH_INCUMBENT_NDCG_AT_10}},
        "search_objective": "Improve seed2026 full-sort validation NDCG@10 over frozen FaGSP using a faithful sparse/spectral residual.",
        "comparison_boundary": "Historical FaGSP result is source-package evidence, not validation of the new adapter runtime.",
    })




def _parent_contract(context):
    from .p4_runtime import actual_execution_contract
    baseline = context.knowledge_base.get("baseline_context", {})
    return actual_execution_contract(baseline["parent_execution_record"])




def _p4_mechanism_text(draft: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field_name in (
        "hypothesis",
        "mechanism_change",
        "minimal_testable_wedge",
    ):
        value = draft.get(field_name)
        if isinstance(value, str):
            values.append(value)
    for field_name in ("causal_chain",):
        value = draft.get(field_name)
        if isinstance(value, (tuple, list)):
            values.extend(str(item) for item in value if isinstance(item, str))
    resolution_facts = draft.get("resolution_facts")
    if isinstance(resolution_facts, Mapping):
        capability_diff = resolution_facts.get("capability_diff")
        if isinstance(capability_diff, (tuple, list)):
            values.extend(
                str(item) for item in capability_diff if isinstance(item, str)
            )
    return "\n".join(values).lower()


def _validate_p4_provider_draft(draft: Mapping[str, Any]) -> None:
    from recclaw_core.research_line.p4_runtime import p4_fit_mode, p4_fit_input_contract
    try:
        mechanism_config = json.loads(draft.get("mechanism_config_json", "{}"))
        if not isinstance(mechanism_config, dict) or "p4_fit_mode" not in mechanism_config:
            raise ValueError("new P4 proposals must explicitly declare p4_fit_mode in mechanism_config_json")
        p4_fit_mode(mechanism_config)
        p4_fit_input_contract(mechanism_config)
    except (TypeError, ValueError) as error:
        raise OpenSpecProjectionError(str(error)) from error
    for field_name in (
        "research_question",
        "hypothesis",
        "minimal_testable_wedge",
        "mechanism_change",
        "competing_explanation",
        "mechanism_off_definition",
        "matched_control_requirement",
        "falsifier",
        "resource_hypothesis",
        "high_change_justification",
    ):
        value = draft.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise OpenSpecProjectionError(
                f"P4 proposal {field_name} must be non-empty"
            )

    def require_text_array(
        field_name: str,
        *,
        minimum: int,
        maximum: int | None = None,
        unique: bool = False,
    ) -> tuple[str, ...]:
        value = draft.get(field_name)
        if not isinstance(value, (tuple, list)):
            raise OpenSpecProjectionError(f"P4 proposal {field_name} must be an array")
        rows = tuple(value)
        if len(rows) < minimum or (maximum is not None and len(rows) > maximum):
            raise OpenSpecProjectionError(f"P4 proposal {field_name} has invalid cardinality")
        if any(not isinstance(row, str) or not row.strip() for row in rows):
            raise OpenSpecProjectionError(f"P4 proposal {field_name} has an empty entry")
        if unique and len(set(rows)) != len(rows):
            raise OpenSpecProjectionError(f"P4 proposal {field_name} must be unique")
        return rows

    require_text_array("causal_chain", minimum=1)
    require_text_array("discriminative_predictions", minimum=1)
    require_text_array("implementation_requirements", minimum=1)
    require_text_array("expected_evidence", minimum=1)
    require_text_array("compatibility_requirements", minimum=4, maximum=4, unique=True)

    resolution_facts = draft.get("resolution_facts")
    if not isinstance(resolution_facts, Mapping):
        raise OpenSpecProjectionError("P4 proposal resolution_facts must be an object")
    for field_name, minimum, maximum, unique in (
        ("capability_diff", 1, None, False),
        ("high_change_dimensions", 1, None, True),
        ("required_dependencies", 0, None, True),
    ):
        value = resolution_facts.get(field_name)
        if not isinstance(value, (tuple, list)):
            raise OpenSpecProjectionError(
                f"P4 proposal resolution_facts.{field_name} must be an array"
            )
        rows = tuple(value)
        if len(rows) < minimum or (maximum is not None and len(rows) > maximum):
            raise OpenSpecProjectionError(
                f"P4 proposal resolution_facts.{field_name} has invalid cardinality"
            )
        if any(not isinstance(row, str) or not row.strip() for row in rows):
            raise OpenSpecProjectionError(
                f"P4 proposal resolution_facts.{field_name} has an empty entry"
            )
        if unique and len(set(rows)) != len(rows):
            raise OpenSpecProjectionError(
                f"P4 proposal resolution_facts.{field_name} must be unique"
            )
    required_budget = resolution_facts.get("required_budget")
    if not isinstance(required_budget, Mapping) or any(
        isinstance(required_budget.get(field_name), bool)
        or not isinstance(required_budget.get(field_name), int)
        or required_budget[field_name] < 1
        for field_name in (
            "implementation_token_ceiling",
            "qualification_gpu_minutes",
            "qualification_wall_minutes",
        )
    ):
        raise OpenSpecProjectionError("P4 proposal required_budget must be positive")

    if draft.get("closest_parent") != P4_EXECUTABLE_PARENT_ID:
        raise OpenSpecProjectionError(
            "P4 v1 proposals must derive from the frozen FaGSP parent"
        )
    if draft.get("realization_mode") != "PARENT_PRESERVING":
        raise OpenSpecProjectionError(
            "P4 v1 proposals must use PARENT_PRESERVING realization"
        )
    contract = draft.get("execution_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "capability_family",
        "model",
        "base_model_config",
        "config",
    }:
        raise OpenSpecProjectionError("P4 execution contract is incomplete")
    capability_family = contract.get("capability_family")
    if (
        not isinstance(capability_family, str)
        or not capability_family.startswith("P4_")
        or contract.get("model") != "FreshCandidateModel"
        or contract.get("base_model_config") != "P4SparseSpectral"
        or contract.get("config") != {}
    ):
        raise OpenSpecProjectionError(
            "P4 execution contract does not name the dedicated sparse-spectral runtime"
        )

    mechanism_text = _p4_mechanism_text(draft)
    p4_markers = (
        "spectral",
        "singular",
        "frequency",
        "filter",
        "interaction matrix",
        "gram",
        "sparse",
        "low-pass",
        "operator",
    )
    if not any(marker in mechanism_text for marker in p4_markers):
        raise OpenSpecProjectionError(
            "P4 proposal does not declare a sparse/spectral operator mechanism"
        )


def _coerce_spec(
    result: Mapping[str, Any],
    *,
    producer_role: str,
    context: Any,
    bindings: Mapping[str, Any],
) -> tuple[OpenResearchSpecV1, Mapping[str, Any]]:
    draft = dict(result)
    _validate_p4_provider_draft(draft)
    mechanism_config = json.loads(draft.pop("mechanism_config_json"))
    # Only newly authored implementation boundaries use the fitting/scoring ABI.
    # Previously materialized packages and their scientific records stay intact.
    mechanism_config["p4_hook_abi"] = 2
    mechanism_delta = json.dumps(mechanism_config)
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import BUDGET_LIMITS
    # Qualification is a machine-owned mini-run, not the researcher's estimate
    # of full training cost. Keep the original estimate in outcome provenance.
    draft["resolution_facts"] = {
        **dict(draft["resolution_facts"]),
        "required_budget": dict(BUDGET_LIMITS),
    }
    # No observed failure means a prospective hypothesis, regardless of the
    # role's label. This never invents a diagnostic or changes its mechanism.
    if draft["observed_failure_mode"] in (None, "NOT_OBSERVED"):
        draft["idea_mode"] = "FRONTIER_HYPOTHESIS"
    bindings = {
        **dict(bindings),
        "compatibility_requirements": tuple(draft["compatibility_requirements"]),
        "implementation_requirements": (
            *draft["implementation_requirements"],
            "fit_residual(parameters, inputs) and residual_scores(coefficients, state, parameters, inputs)",
            "isolated declared fitting view, followed by full-train final scoring inputs",
            "machine-owned FaGSP parent addition, coefficient/state registration, static cache and actual zero-action intervention",
        ),
    }
    # Provider must first prove that the proposal is genuinely P4. Only then
    # may the profile inject frozen identities that the model cannot control.
    draft["implementation_requirements"] = tuple(
        bindings["implementation_requirements"]
    )
    draft["compatibility_requirements"] = tuple(
        bindings["compatibility_requirements"]
    )
    from .p4_runtime import merge_mechanism_delta
    draft["execution_contract"] = merge_mechanism_delta(_parent_contract(context), mechanism_delta)
    spec, facts = project_open_producer_draft(
        draft,
        bindings=bindings,
        strict_resolution_contract=True,
    )
    if spec.producer_role != producer_role:
        raise OpenSpecProjectionError("P4 spec producer_role drift")
    if spec.protocol_ref != context.protocol_ref or spec.protocol_digest != context.protocol_digest:
        raise OpenSpecProjectionError("P4 spec protocol drift")
    return spec, facts


def _mechanism_program(
    spec: OpenResearchSpecV1,
    facts: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract = spec.execution_contract
    if contract is None:
        raise ValueError("P4 mechanism lacks its inherited execution contract")
    return canonical_value(
        {
            "schema": "recclaw.p4-sparse-spectral.mechanism-program.v1",
            "profile_id": P4_PROFILE_ID,
            "protocol_ref": P4_PROTOCOL_REF,
            "protocol_digest": P4_PROTOCOL_DIGEST,
            "producer_role": spec.producer_role,
            "hypothesis": spec.hypothesis,
            "mechanism_change": spec.mechanism_change,
            "mechanism_axis": "SPARSE_SPECTRAL_OPERATOR",
            "closest_parent": spec.closest_parent,
            "execution_contract": contract,
            "resolution_facts": canonical_value(dict(facts)),
        }
    )


def _candidate_identity(
    spec: OpenResearchSpecV1,
    program: Mapping[str, Any],
) -> tuple[str, str, str]:
    semantic_digest = sha256_digest(
        {
            "profile_id": P4_PROFILE_ID,
            "program": program,
        }
    )
    candidate_id = "p4_" + semantic_digest[:24]
    return (
        candidate_id,
        f"p4-sparse-spectral-mechanism:{candidate_id}",
        semantic_digest,
    )


def _effective_family_digest(program: Mapping[str, Any]) -> str:
    mechanism_change = program.get("mechanism_change")
    if not isinstance(mechanism_change, str) or not mechanism_change.strip():
        raise ValueError("P4 mechanism program lacks mechanism_change")
    return sha256_digest(
        {
            "space": P4_PROFILE_ID,
            "axis": program.get("mechanism_axis", "SPARSE_SPECTRAL_OPERATOR"),
            "parent": program.get("closest_parent"),
            "mechanism_change": mechanism_change.strip(),
        }
    )


class P4SparseSpectralSearchSpaceAdapterV1:
    adapter_id = P4_ADAPTER_ID
    supported_frozen_profile_kinds = ("OFFLINE_TOPN",)
    compatibility_requirements = (
        "static unordered implicit-feedback recommendation",
        "train-only operator fitting",
        "full-sort NDCG@10 validation selection",
        "no learned sequential, semantic-ID, or diffusion/flow state",
    )

    def __init__(self, parent_execution_record: Mapping[str, Any]):
        self.baseline_context = p4_baseline_context(parent_execution_record)

    def build_provider_research_producer(self, **kwargs):
        from .provider import ProviderResearchProducer, RESEARCH_BL_ICF_PROGRAM_TOTAL_TOKEN_CEILING
        resources = Path(__file__).resolve().parents[1] / "experiments/helix_abc_v1/resources"
        schema = resources / "research_line_p4_sparse_spectral_proposal_response_v1.schema.json"
        prompt = (resources / "research_line_open_spec_proposal_prompt_v1.txt").read_text(encoding="utf-8")
        prompt += "\n\n" + (resources / "research_line_p4_sparse_spectral_proposal_prompt_v1.txt").read_text(encoding="utf-8")
        # The full record also contains the same config and runtime provenance.
        # Expose the actual research config once; keep the record in the adapter.
        parent = self.baseline_context["parent_anchor"]
        prompt += "\nActual parent configuration (inherit omitted config; no guessed defaults):\n" + json.dumps({
            "execution_contract": parent["execution_contract"],
            "paired_metric": parent["paired_metric"],
            "comparison_boundary": self.baseline_context["comparison_boundary"],
        }, sort_keys=True, indent=2)
        # Like the other native mechanism profiles, account for context input
        # separately from the unchanged per-call/per-slot output budget.
        kwargs.setdefault("total_token_ceiling", RESEARCH_BL_ICF_PROGRAM_TOTAL_TOKEN_CEILING)
        kwargs.setdefault("proposal_protocol_requirements", self.compatibility_requirements)
        return ProviderResearchProducer(proposal_template_source=prompt,
            proposal_schema_source=schema, proposal_schema_path=schema, **kwargs)

    def build_provider_implementer(self, **kwargs):
        import inspect
        from ..experiments.helix_abc_v1 import p4_fagsp_parent
        from .provider import ProviderImplementerGateway
        resources = Path(__file__).resolve().parents[1] / "experiments/helix_abc_v1/resources"
        template = (resources / "research_line_p4_sparse_spectral_implementer_prompt_v2.txt").read_text()
        parent_reference = "FROZEN_FAGSP_PARAMETERS = " + repr(dict(p4_fagsp_parent.FROZEN_FAGSP_PARAMETERS)) + "\n\n"
        parent_reference += "\n\n".join(inspect.getsource(function) for function in (
            p4_fagsp_parent._normalize,
            p4_fagsp_parent._spectral_complement_action,
            p4_fagsp_parent.frozen_fagsp_scores,
        ))
        return ProviderImplementerGateway(implementation_template_source=template.replace(
            "{{FROZEN_FAGSP_REFERENCE}}", parent_reference), **kwargs)

    def qualification_unit_check(self, **kwargs):
        from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import validate_p4_sparse_spectral_model_contract
        return validate_p4_sparse_spectral_model_contract

    def initial_profile(self, campaign_id: str) -> SearchExecutableProfileV1:
        return SearchExecutableProfileV1(
            campaign_id=campaign_id,
            profile_ref=P4_PROFILE_ID,
            profile_digest=P4_PROFILE_DIGEST,
            protocol_ref=P4_PROTOCOL_REF,
            protocol_digest=P4_PROTOCOL_DIGEST,
            activation=SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN,
            predecessor_campaign_id=None,
            predecessor_profile_ref=None,
            predecessor_profile_digest=None,
            entries=(),
        )

    def conformance(self, profile: SearchExecutableProfileV1) -> Mapping[str, Any]:
        return canonical_value(
            {
                "adapter_id": self.adapter_id,
                "search_space_id": profile.profile_ref,
                "search_space_digest": profile.profile_digest,
                "protocol_ref": profile.protocol_ref,
                "protocol_digest": profile.protocol_digest,
                "active_executable_capability_count": len(profile.entries),
                "fixed_fallback": False,
                "baseline_pack_digest": sha256_digest(P4_BASELINE_PACK),
                "raw_cross_profile_ndcg_competition": False,
            }
        )

    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1:
        phase = context.get("phase")
        if phase == "PROJECT_PROVIDER_PROPOSAL":
            if not isinstance(primary_binding, Mapping):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_PROVIDER_RESULT_IS_NOT_OPEN_DRAFT",
                )
            try:
                spec, facts = _coerce_spec(
                    primary_binding,
                    producer_role=str(context["producer_role"]),
                    context=context["research_context"],
                    bindings=context["producer_bindings"],
                )
            except (KeyError, TypeError, ValueError) as error:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason=(
                        "P4_OPEN_SPEC_PROJECTION_FAILED:"
                        f"{type(error).__name__}:{error}"
                    ),
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": ProducerOutcome(
                        producer_role=spec.producer_role,
                        context_ref=spec.context_ref,
                        context_digest=spec.context_digest,
                        spec=spec,
                        resolution_facts=facts,
                        provenance={
                            "producer_role": spec.producer_role,
                            "status": "PRODUCED",
                            "search_space_profile": P4_PROFILE_ID,
                            "provider_idea_mode": primary_binding["idea_mode"],
                            "provider_resource_budget_estimate": canonical_value(primary_binding["resolution_facts"]["required_budget"]),
                            "qualification_budget_source": "fresh_r1.BUDGET_LIMITS",
                        },
                    )
                },
            )
        if phase in {"IDENTIFY_INNOVATION", "PREPARE_INNOVATION"}:
            outcome = primary_binding
            if not isinstance(outcome, ProducerOutcome) or outcome.spec is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_INNOVATION_OUTCOME_INCOMPLETE",
                )
            program = _mechanism_program(outcome.spec, outcome.resolution_facts)
            candidate_id, semantic_ref, semantic_digest = _candidate_identity(
                outcome.spec,
                program,
            )
            effective_identity = canonical_value(
                {
                    "effective_experiment_digest": sha256_digest(
                        {
                            "space": P4_PROFILE_ID,
                            "protocol_digest": P4_PROTOCOL_DIGEST,
                            "mechanism_program": program,
                        }
                    ),
                    "effective_family_digest": _effective_family_digest(program),
                    "primitive_ids": ("P4_SPARSE_SPECTRAL_OPERATOR",),
                }
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": outcome,
                    "compiler_binding": None,
                    "mechanism_program": program,
                    "compiled_implementation": None,
                    "semantic_identity_ref": semantic_ref,
                    "semantic_identity_digest": semantic_digest,
                    "effective_identity": effective_identity,
                    "candidate_id": candidate_id,
                    "mechanism_id": "P4_SPARSE_SPECTRAL_OPERATOR",
                    "parent_available": (
                        outcome.spec.closest_parent == P4_EXECUTABLE_PARENT_ID
                    ),
                },
            )
        if phase == "PACKAGE_QUALIFIED_INNOVATION":
            prepared = context.get("prepared_innovation")
            materialized = context.get("materialized")
            capability = context.get("capability")
            utility = context.get("utility_features")
            evidence = context.get("feature_evidence")
            if (
                not isinstance(prepared, Mapping)
                or materialized is None
                or capability is None
                or not isinstance(utility, Mapping)
                or not isinstance(evidence, Mapping)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE",
                )
            outcome = prepared["producer_outcome"]
            if not isinstance(outcome, ProducerOutcome):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_QUALIFIED_OUTCOME_INVALID",
                )
            package = materialized.package
            if (
                package.package_id != capability.candidate_package_ref
                or package.digest != capability.candidate_package_digest
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_QUALIFIED_PACKAGE_IDENTITY_DRIFT",
                )
            candidate = P4SparseSpectralCandidateV1(
                spec=outcome.spec,
                candidate_id=str(prepared["candidate_id"]),
                producer_role=outcome.producer_role,
                mechanism_id=str(prepared["mechanism_id"]),
                mechanism_axis="SPARSE_SPECTRAL_OPERATOR",
                mechanism_program=prepared["mechanism_program"],
                parent_candidate_id=outcome.spec.closest_parent if outcome.spec else None,
                utility_features=SearchUtilityFeaturesV1(**dict(utility)),
                feature_evidence=RouterFeatureEvidenceV1(**dict(evidence)),
                semantic_identity_ref=str(prepared["semantic_identity_ref"]),
                semantic_identity_digest=str(prepared["semantic_identity_digest"]),
                mechanism_program_digest=sha256_digest(prepared["mechanism_program"]),
                candidate_package_ref=capability.candidate_package_ref,
                candidate_package_digest=capability.candidate_package_digest,
                execution_contract=outcome.spec.execution_contract,
                capability_ref=capability.capability_id,
                capability_digest=capability.digest,
                executable_entrypoint=package.executable_entrypoint,
                qualification_receipt_ref=capability.qualification_receipt_ref,
                qualification_receipt_digest=capability.qualification_receipt_digest,
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"search_candidate": candidate},
            )
        if phase == "MATERIALIZE_DISCOVERY":
            outcome = primary_binding
            resolution = context.get("resolution")
            profile = context.get("execution_profile")
            proposal = getattr(outcome, "source_proposal", None)
            capability_ref = getattr(resolution, "resolved_current_capability_ref", None)
            if proposal is None or profile is None or capability_ref is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="P4_DISCOVERY_BINDING_CONTEXT_INCOMPLETE",
                )
            entry = profile.entry(capability_ref)
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "native_binding": SearchCandidateBindingV1(
                        proposal=proposal,
                        capability_ref=entry.capability_ref,
                        capability_digest=entry.capability_digest,
                        executable_entrypoint=entry.executable_entrypoint,
                        entry_origin=entry.origin,
                        mechanism_semantics_digest=entry.semantic_identity_digest,
                    )
                },
            )
        if phase in {"DECLARE_FOLLOWUP", "MATCH_PROPOSAL", "MATERIALIZE_VERIFICATION"}:
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                reason="P4_CONTROL_BINDING_REQUIRES_EXPLICIT_PROFILE_CANDIDATE",
            )
        return ConfirmationResolutionV1(
            ConfirmationResolutionKindV1.UNSUPPORTED,
            reason="P4_CONFIRMATION_PHASE_UNSUPPORTED",
        )

    def validate_execution_binding(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> None:
        if binding.adapter_id != self.adapter_id:
            raise ValueError("P4 adapter binding id mismatch")
        self.execution_recipe(binding)

    def effective_identity(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        proposal = binding.native_binding.proposal
        program = getattr(proposal, "mechanism_program", None)
        if not isinstance(program, Mapping):
            raise ValueError("P4 binding lacks mechanism_program")
        return canonical_value(
            {
                "effective_experiment_digest": sha256_digest(
                    {
                        "space": P4_PROFILE_ID,
                        "protocol_digest": P4_PROTOCOL_DIGEST,
                        "mechanism_program": program,
                    }
                ),
                "effective_family_digest": _effective_family_digest(program),
                "primitive_ids": ("P4_SPARSE_SPECTRAL_OPERATOR",),
            }
        )

    def execution_recipe(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        native = binding.native_binding
        proposal = native.proposal
        context = binding.execution_context
        qualified_execution = context.get("qualified_execution")
        if native.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
            if not isinstance(qualified_execution, Mapping):
                raise ValueError("P4 qualified binding lacks execution evidence")
            recipe = dict(qualified_execution)
        elif isinstance(proposal, P4SparseSpectralCandidateV1):
            contract = proposal.mechanism_program["execution_contract"]
            recipe = {
                "capability_family": contract["capability_family"],
                "model": contract["model"],
                "base_model_config": contract["base_model_config"],
                "config": contract["config"],
                "entrypoint_source_sha256": sha256_digest(contract),
            }
        else:
            raise ValueError("unsupported P4 fixed binding")
        recipe.update(
            {
                "capability_ref": native.capability_ref,
                "capability_digest": native.capability_digest,
                "profile_ref": context["profile"].profile_ref,
                "profile_digest": context["profile"].profile_digest,
                "entrypoint": native.executable_entrypoint,
                "mechanism_id": proposal.mechanism_id,
                "mechanism_semantics_digest": native.mechanism_semantics_digest,
                "dataset": COMMON_DATASET,
                "split": P4_SPARSE_SPECTRAL_SPLIT,
                "evaluator": P4_SPARSE_SPECTRAL_EVALUATOR,
                "execution_role": "CANDIDATE",
                "p4_protocol_ref": P4_PROTOCOL_REF,
                "p4_protocol_digest": P4_PROTOCOL_DIGEST,
            }
        )
        normalized = canonical_value(recipe)
        validate_execution_recipe(normalized)
        return normalized


def default_p4_sparse_spectral_adapter(parent_execution_record: Mapping[str, Any]) -> P4SparseSpectralSearchSpaceAdapterV1:
    return P4SparseSpectralSearchSpaceAdapterV1(parent_execution_record)


__all__ = [
    "P4_ADAPTER_ID",
    "P4_BASELINE_PACK",
    "P4_FAGSP_COMPARATOR_DIGEST",
    "P4_FAGSP_COMPARATOR_IDENTITY",
    "P4_FAGSP_FINAL_COMPARATOR_DIGEST",
    "P4_FAGSP_FINAL_COMPARATOR_IDENTITY",
    "P4_FAGSP_PARENT_EXECUTION_DIGEST",
    "P4_FAGSP_PARENT_EXECUTION_IDENTITY",
    "P4_FAGSP_RUNNER_RELATIVE_PATH",
    "P4_FAGSP_SEARCH_INCUMBENT_DIGEST",
    "P4_FAGSP_SEARCH_INCUMBENT_IDENTITY",
    "P4_FINAL_COMPARATOR_REF",
    "P4_FINAL_COMPARATOR_TEST_NDCG_AT_10_MEAN",
    "P4_EXECUTABLE_PARENT_ID",
    "P4_PARENT_REGISTRY",
    "P4_PRIMARY_COMPARATOR_NDCG_AT_10",
    "P4_PRIMARY_COMPARATOR_REF",
    "P4_PROFILE_DIGEST",
    "P4_PROFILE_ID",
    "P4_PROTOCOL",
    "P4_PROTOCOL_DIGEST",
    "P4_PROTOCOL_REF",
    "P4_SEARCH_INCUMBENT_NDCG_AT_10",
    "P4_SEARCH_INCUMBENT_REF",
    "P4SparseSpectralCandidateV1",
    "P4SparseSpectralSearchSpaceAdapterV1",
    "default_p4_sparse_spectral_adapter",
    "p4_baseline_context",
    "p4_frozen_profile_ref",
    "validate_p4_profile_assets",
]
