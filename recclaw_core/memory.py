"""Typed memory ingestion and retrieval for Candidate Foundry v0.1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from recclaw_core.contracts import NATIVE_BPR_RANKCUT_MEMORY_ID, load_yaml, source_rows


ADVANCE_ACTIONS = {"implementation_plan", "activation_smoke", "metric_smoke", "formal_candidate"}


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}


def _memory_text(memory: Mapping[str, Any]) -> str:
    fields: List[str] = [
        str(memory.get("memory_id", "")),
        str(memory.get("family", "")),
        str(memory.get("mechanism", "")),
        str(memory.get("observed_failure", "")),
        str(memory.get("outcome", "")),
    ]
    for key in ("retrieval_terms", "allowed_effect", "future_revival_condition", "what_to_avoid"):
        value = memory.get(key, [])
        if isinstance(value, Sequence) and not isinstance(value, str):
            fields.extend(str(item) for item in value)
        else:
            fields.append(str(value))
    return " ".join(fields)


def ingest_bpr_rankcut_memory(artifact_dir: Path) -> Dict[str, Any]:
    """Create native diagnostic memory from controlled Phase 9.5-style artifacts."""

    adjudication_path = artifact_dir / "final_phase9_adjudication.yaml"
    comparison_path = artifact_dir / "disabled_equivalence_comparison.yaml"
    typed_memory_path = artifact_dir / "typed_memory_update.yaml"
    adjudication = load_yaml(adjudication_path)
    comparison = load_yaml(comparison_path)
    typed_memory = load_yaml(typed_memory_path)

    if adjudication.get("final_outcome") != "mechanism_negative_activation_metric_mismatch":
        raise ValueError("BPR rank-cut memory requires mechanism_negative_activation_metric_mismatch")
    if comparison.get("disabled_metric_equivalent") is not True:
        raise ValueError("BPR rank-cut memory requires disabled metric equivalence")

    return {
        "memory_id": NATIVE_BPR_RANKCUT_MEMORY_ID,
        "memory_type": "diagnostic_negative_memory",
        "family": "BPR",
        "mechanism": "rank_cut_margin",
        "candidate_id": typed_memory.get("candidate_id"),
        "parent_anchor": typed_memory.get("parent_anchor"),
        "outcome": adjudication.get("final_outcome"),
        "observed_failure": "activation signal did not translate to full-sort ranking metrics",
        "active_metric_delta_summary": adjudication.get("active_phase9_metric_deltas", {}),
        "disabled_metric_equivalence": {
            "disabled_metric_equivalent": comparison.get("disabled_metric_equivalent"),
            "metric_deltas_disabled_minus_parent": comparison.get("metric_deltas_disabled_minus_parent", {}),
            "tolerance_by_metric": comparison.get("tolerance_by_metric", {}),
        },
        "retrieval_terms": [
            "bpr",
            "rankcut",
            "rank cut",
            "rank surrogate",
            "batch local cutoff",
            "top-k surrogate",
            "activation metric mismatch",
            "full-sort metric regression",
        ],
        "allowed_effect": [
            "penalize future batch-local rank-cut performance candidates",
            "allow diagnostic reference",
            "require global top-k alignment evidence for revival",
        ],
        "what_to_avoid": [
            "do not promote batch-local rank-cut as a performance candidate",
            "do not claim search-quality improvement",
            "do not claim metric improvement",
        ],
        "future_revival_condition": [
            "only revisit through a globally aligned top-k surrogate or exposure-aware objective",
        ],
        "claim_ceiling": "diagnostic negative memory only; no metric, formal, M5, runtime-kernel, or search-quality claim",
        "source_policy": {
            "controlled_structured_artifacts_only": True,
            "uses_codex_chat": False,
            "uses_lablog_prose": False,
            "uses_architecture_memo": False,
        },
        "source_artifacts": source_rows(
            [adjudication_path, comparison_path, typed_memory_path],
            relative_to=artifact_dir.parent,
        ),
    }


def retrieve_memory(
    query: str,
    memories: Iterable[Mapping[str, Any]],
    *,
    min_overlap: int = 2,
) -> List[Dict[str, Any]]:
    """Return memory ids with deterministic token-overlap scores."""

    query_tokens = _tokens(query)
    rows: List[Dict[str, Any]] = []
    for memory in memories:
        score = len(query_tokens & _tokens(_memory_text(memory)))
        if score >= min_overlap:
            rows.append({"memory_id": memory.get("memory_id"), "score": score})
    return sorted(rows, key=lambda row: (-int(row["score"]), str(row["memory_id"])))
