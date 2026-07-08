"""Candidate card validation for RecClaw Candidate Foundry v0.1."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from recclaw_core.contracts import CANDIDATE_REQUIRED_FIELDS, POLICY_FEATURES


def validate_candidate_card(card: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(CANDIDATE_REQUIRED_FIELDS - set(card))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")

    features = card.get("policy_features", {})
    if not isinstance(features, Mapping):
        errors.append("policy_features:not_mapping")
        return errors

    for key in sorted(POLICY_FEATURES):
        if key not in features:
            errors.append(f"missing_policy_feature:{key}")
            continue
        value = features[key]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"policy_feature_out_of_range:{key}")

    if "search-quality" not in str(card.get("claim_ceiling", "")).lower() and "search quality" not in str(
        card.get("claim_ceiling", "")
    ).lower():
        errors.append("claim_ceiling_missing_search_quality_boundary")

    return errors


def validate_candidate_cards(cards: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    card_list = list(cards)
    for card in card_list:
        rows.append({"candidate_id": card.get("candidate_id"), "errors": validate_candidate_card(card)})
    invalid = [row for row in rows if row["errors"]]
    return {
        "status": "candidate_validation_passed" if not invalid else "candidate_validation_failed",
        "candidate_count": len(card_list),
        "valid_candidate_count": len(card_list) - len(invalid),
        "rows": rows,
    }
