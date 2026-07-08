"""Deterministic Candidate Foundry policy ranking."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from recclaw_core.contracts import DEFAULT_POLICY_WEIGHTS, NATIVE_BPR_RANKCUT_MEMORY_ID


def _base_score(features: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return sum(float(features.get(key, 0.0)) * weight for key, weight in weights.items())


def _memory_adjustment(card: Mapping[str, Any], memory_ids: set[str]) -> tuple[float, List[str], bool]:
    features = card.get("policy_features", {})
    is_rankcut_like = bool(features.get("rankcut_like"))
    global_topk_alignment = bool(features.get("global_topk_alignment"))
    eligible = True
    adjustment = 0.0
    reasons: List[str] = []
    if NATIVE_BPR_RANKCUT_MEMORY_ID in memory_ids and is_rankcut_like and not global_topk_alignment:
        adjustment -= 0.65
        eligible = False
        reasons.append("phase10_memory_blocks_batch_local_rankcut_repeat")
    elif NATIVE_BPR_RANKCUT_MEMORY_ID in memory_ids and is_rankcut_like:
        adjustment -= 0.22
        reasons.append("phase10_memory_downweights_rankcut_family_until_global_alignment_is_proven")
    if NATIVE_BPR_RANKCUT_MEMORY_ID in memory_ids and card.get("family") == "LightGCN":
        adjustment += 0.04
        reasons.append("phase10_memory_shifts_budget_toward_lightgcn_opportunity")
    if card.get("candidate_role") == "diagnostic_info_value_candidate":
        adjustment += 0.05
        reasons.append("keep_one_non_bpr_information_value_branch")
    return adjustment, reasons, eligible


def rank_candidates(
    cards: Iterable[Mapping[str, Any]],
    memories: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    weights = weights or DEFAULT_POLICY_WEIGHTS
    memory_ids = {str(memory.get("memory_id")) for memory in memories}
    scored: List[Dict[str, Any]] = []
    for card in cards:
        features = card.get("policy_features", {})
        base = _base_score(features, weights)
        adjustment, reasons, eligible = _memory_adjustment(card, memory_ids)
        scored.append(
            {
                "candidate_id": card.get("candidate_id"),
                "family": card.get("family"),
                "candidate_role": card.get("candidate_role"),
                "base_score": round(base, 6),
                "memory_adjustment": round(adjustment, 6),
                "final_score": round(base + adjustment, 6),
                "selection_eligible": eligible,
                "memory_effect_reasons": reasons,
            }
        )
    ranked = sorted(scored, key=lambda row: (-float(row["final_score"]), str(row["candidate_id"])))
    return {"status": "deterministic_policy_scored", "ranked_candidates": ranked}


def select_candidate_queue(ranked: Iterable[Mapping[str, Any]], *, limit: int = 2) -> Dict[str, Any]:
    rows = [row for row in ranked if row.get("selection_eligible") is True]
    opportunity = [
        row for row in rows if row.get("family") == "LightGCN" and row.get("candidate_role") == "opportunity_candidate"
    ]
    diagnostic = [row for row in rows if str(row.get("candidate_role", "")).startswith("diagnostic")]
    selected: List[Mapping[str, Any]] = []
    if opportunity:
        selected.append(max(opportunity, key=lambda row: float(row["final_score"])))
    if diagnostic:
        first_family = selected[0].get("family") if selected else None
        diverse = [row for row in diagnostic if row.get("family") != first_family] or diagnostic
        selected.append(max(diverse, key=lambda row: float(row["final_score"])))
    return {
        "status": "selected_candidate_queue_ready",
        "selected_candidates": [
            {
                "rank": index,
                "candidate_id": row.get("candidate_id"),
                "family": row.get("family"),
                "candidate_role": row.get("candidate_role"),
                "final_score": row.get("final_score"),
            }
            for index, row in enumerate(selected[:limit], start=1)
        ],
    }
