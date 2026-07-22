#!/usr/bin/env python3
"""Research-ability line utilities for RecClaw.

This module makes the algorithm-search side explicit:

- Candidate Producers: role-specific proposal sources.
- Research Router: deterministic scoring for proposals and runnable options.
- Search Memory: compact statistics over prior trials, blockers, and producers.
- Meta-Research Controller: offline, advisory updates for later strategy changes.

The code is intentionally side-effect free so it can be reused by agent.py,
propose_candidate.py, and comparison/planning scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a project dependency.
    yaml = None  # type: ignore[assignment]


ALGORITHM_ACTION_TYPES = {
    "local_loss",
    "aggregation",
    "regularization",
    "sampling_wrapper",
    "negative_sampling",
    "pairwise_loss",
    "graph_propagation",
    "graph_augmentation",
    "layer_interaction",
    "auxiliary_loss",
    "rank_aware_loss",
}
ACTION_INFORMATION_PRIOR = {
    "rank_aware_loss": 0.24,
    "auxiliary_loss": 0.22,
    "layer_interaction": 0.21,
    "graph_augmentation": 0.20,
    "negative_sampling": 0.19,
    "pairwise_loss": 0.18,
    "local_loss": 0.17,
    "regularization": 0.13,
    "aggregation": 0.12,
    "sampling_wrapper": 0.11,
    "parameter_tuning": 0.04,
    "posthoc_rerank": 0.02,
}
COMPLEXITY_COST = {"low": 0.02, "medium": 0.08, "high": 0.17}
RUNTIME_RISK_KEYWORDS = ("heavy", "expensive", "slow", "unstable", "large overhead")
BLOCKER_STATUS = {
    "crash",
    "failed",
    "error",
    "implementation_failed",
    "implementation_rejected",
    "implemented_but_smoke_failed",
}


@dataclass(frozen=True)
class ProducerDirective:
    producer_id: str
    role: str
    proposal_mode: str
    target_count: int
    priority_families: list[str] = field(default_factory=list)
    preferred_actions: list[str] = field(default_factory=list)
    avoid_families: list[str] = field(default_factory=list)
    instruction: str = ""


@dataclass(frozen=True)
class RouteDecision:
    candidate_id: str
    score: float
    factors: dict[str, float]
    decision: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": round(self.score, 6),
            "factors": {key: round(value, 6) for key, value in sorted(self.factors.items())},
            "decision": self.decision,
            "reason": self.reason,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    return parsed if isinstance(parsed, dict) else {}


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_signature(raw: Any) -> str:
    return str(raw or "").replace(" ", "").strip()


def family_of(record: dict[str, Any]) -> str:
    return str(record.get("parent_candidate_id") or record.get("candidate_id") or "unknown")


def mechanism_tokens(record: dict[str, Any]) -> list[str]:
    raw = record.get("mechanism_composition")
    if isinstance(raw, list):
        return [str(item).lower().strip() for item in raw if str(item).strip()]
    text = " ".join(
        str(record.get(key) or "")
        for key in ("mechanism", "candidate_id", "hypothesis", "novelty_claim")
    ).lower()
    tokens = []
    for token in (
        "hard_negative",
        "margin",
        "tail",
        "popularity",
        "norm",
        "residual",
        "edge_dropout",
        "alignment",
        "rank",
        "gate",
        "sampling",
    ):
        if token in text:
            tokens.append(token)
    return tokens


def family_matches(candidate_family: str, target_family: str) -> bool:
    candidate_family = str(candidate_family or "")
    target_family = str(target_family or "")
    return (
        candidate_family == target_family
        or candidate_family.startswith(f"{target_family}_")
        or target_family.startswith(f"{candidate_family}_")
    )


def extract_result_metric(record: dict[str, Any], metric: str) -> float | None:
    result = record.get("result")
    if isinstance(result, dict):
        value = to_float(result.get(metric))
        if value is not None:
            return value
    return to_float(record.get(metric))


def build_search_memory(
    rows: list[dict[str, Any]],
    *,
    metric: str = "ndcg@10",
    anchor_families: list[str] | None = None,
) -> dict[str, Any]:
    anchor_families = list(anchor_families or [])
    trial_rows = [row for row in rows if not row.get("event")]
    event_rows = [row for row in rows if row.get("event")]
    family_stats: dict[str, dict[str, Any]] = {}
    producer_stats: dict[str, dict[str, Any]] = {}
    duplicate_signatures: set[str] = set()
    seen_signatures: set[str] = set()
    blocker_counts: dict[str, int] = {}
    best_metric: float | None = None
    best_family = ""

    for row in trial_rows:
        family = family_of(row)
        stats = family_stats.setdefault(
            family,
            {
                "family": family,
                "runs": 0,
                "keeps": 0,
                "revises": 0,
                "discards": 0,
                "crashes": 0,
                "best": None,
                "last_status": "",
                "high_potential": False,
            },
        )
        stats["runs"] += 1
        decision = str(row.get("decision") or "").lower()
        stats["keeps"] += int(decision == "keep")
        stats["revises"] += int(decision == "revise")
        stats["discards"] += int(decision == "discard")
        status = str(row.get("status") or "").lower()
        stats["crashes"] += int(decision == "crash" or status in BLOCKER_STATUS)
        stats["last_status"] = status
        value = extract_result_metric(row, metric)
        if value is not None:
            current_best = to_float(stats.get("best"))
            if current_best is None or value > current_best:
                stats["best"] = round(value, 6)
            if best_metric is None or value > best_metric:
                best_metric = value
                best_family = family
        signature = normalize_signature(row.get("parameter_signature") or row.get("execution_signature"))
        if signature:
            if signature in seen_signatures:
                duplicate_signatures.add(signature)
            seen_signatures.add(signature)
        producer = str(row.get("producer_id") or row.get("proposal_source") or "")
        if producer:
            producer_row = producer_stats.setdefault(
                producer,
                {"producer_id": producer, "trials": 0, "keeps": 0, "crashes": 0, "best": None},
            )
            producer_row["trials"] += 1
            producer_row["keeps"] += int(decision == "keep")
            producer_row["crashes"] += int(decision == "crash" or status in BLOCKER_STATUS)
            value = extract_result_metric(row, metric)
            if value is not None:
                current_best = to_float(producer_row.get("best"))
                if current_best is None or value > current_best:
                    producer_row["best"] = round(value, 6)

    for event in event_rows:
        status = str(event.get("status") or event.get("event") or "").lower()
        if not any(token in status for token in ("rejected", "failed", "crash", "blocker", "skipped")):
            continue
        reason_parts = []
        for key in ("reason", "errors", "review_reasons", "expected_failure_mode"):
            value = event.get(key)
            if isinstance(value, list):
                reason_parts.extend(str(item) for item in value[:3])
            elif value:
                reason_parts.append(str(value))
        signature = "; ".join(reason_parts)[:180] or status
        blocker_counts[signature] = blocker_counts.get(signature, 0) + 1

    for family, stats in family_stats.items():
        best = to_float(stats.get("best"))
        stats["high_potential"] = (
            stats["keeps"] > 0
            or stats["revises"] >= 2
            or any(family_matches(family, anchor) for anchor in anchor_families)
            or (best is not None and best >= 0.274)
        )

    frozen_families = [
        family
        for family, stats in family_stats.items()
        if int(stats.get("crashes") or 0) >= 2 and not bool(stats.get("high_potential"))
    ]
    promising_families = [
        family
        for family, stats in sorted(
            family_stats.items(),
            key=lambda item: (
                to_float(item[1].get("best")) if to_float(item[1].get("best")) is not None else -1,
                int(item[1].get("keeps") or 0),
                int(item[1].get("revises") or 0),
            ),
            reverse=True,
        )
        if bool(stats.get("high_potential"))
    ]
    return {
        "metric": metric,
        "trial_count": len(trial_rows),
        "event_count": len(event_rows),
        "best_metric": round(best_metric, 6) if best_metric is not None else None,
        "best_family": best_family,
        "family_stats": family_stats,
        "producer_stats": producer_stats,
        "duplicate_signatures": sorted(duplicate_signatures),
        "seen_signatures": sorted(seen_signatures),
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))),
        "frozen_families": sorted(frozen_families),
        "promising_families": promising_families[:10],
    }


def _stage(policy: dict[str, Any], name: str) -> dict[str, Any]:
    stages = policy.get("search_stages") if isinstance(policy.get("search_stages"), dict) else {}
    stage = stages.get(name) if isinstance(stages.get(name), dict) else {}
    return stage


def _divide_budget(total: int, weights: list[tuple[str, float]]) -> dict[str, int]:
    total = max(1, int(total))
    positive = [(name, max(0.0, weight)) for name, weight in weights]
    weight_sum = sum(weight for _, weight in positive) or 1.0
    if total >= sum(1 for _, weight in positive if weight > 0):
        allocation = {name: (1 if weight > 0 else 0) for name, weight in positive}
        remaining = total - sum(allocation.values())
        weighted = {
            name: max(0, int(remaining * weight / weight_sum))
            for name, weight in positive
        }
        for name, count in weighted.items():
            allocation[name] += count
    else:
        allocation = {name: max(0, int(total * weight / weight_sum)) for name, weight in positive}
    while sum(allocation.values()) < total:
        for name, _ in positive:
            allocation[name] += 1
            if sum(allocation.values()) >= total:
                break
    while sum(allocation.values()) > total:
        for name, _ in sorted(positive, key=lambda item: allocation[item[0]], reverse=True):
            if allocation[name] > 0:
                allocation[name] -= 1
            if sum(allocation.values()) <= total:
                break
    return allocation


def build_producer_directives(
    *,
    search_policy: dict[str, Any] | None,
    search_memory: dict[str, Any] | None,
    proposal_count: int,
    mode: str,
) -> list[ProducerDirective]:
    policy = search_policy or {}
    memory = search_memory or {}
    algorithm_stage = _stage(policy, "algorithm_discovery")
    repair_stage = _stage(policy, "repair")
    explore_stage = _stage(policy, "explore")
    exploit_stage = _stage(policy, "exploit")
    algorithm_first = policy.get("algorithm_first") if isinstance(policy.get("algorithm_first"), dict) else {}
    priority_families = [
        str(item)
        for item in (
            algorithm_stage.get("priority_families")
            or algorithm_first.get("anchor_families")
            or memory.get("promising_families")
            or []
        )
        if str(item)
    ]
    avoid_families = [str(item) for item in memory.get("frozen_families", []) if str(item)]
    plateau = memory.get("plateau_analysis") if isinstance(memory.get("plateau_analysis"), dict) else {}
    if plateau.get("plateau_detected"):
        priority_families = [
            str(item)
            for item in plateau.get("forced_exploration_families", [])
            if str(item)
        ] or priority_families
        avoid_families.extend(
            str(item)
            for item in plateau.get("local_repair_capped_families", [])
            if str(item)
        )

    if mode == "conservative":
        weights = [("heuristic_tuning", 1.0)]
    elif mode == "explore":
        weights = [("llm_algorithm", 0.55), ("heuristic_algorithm", 0.35), ("novelty_spec", 0.10)]
    elif mode == "algorithm_first":
        weights = [("llm_algorithm", 0.55), ("heuristic_algorithm", 0.30), ("repair_ablation", 0.10), ("heuristic_tuning", 0.05)]
    else:
        weights = [("llm_algorithm", 0.35), ("heuristic_algorithm", 0.25), ("repair_ablation", 0.20), ("heuristic_tuning", 0.20)]
    allocation = _divide_budget(proposal_count, weights)
    directives = []
    role_map = {
        "llm_algorithm": ("mechanism_discovery", "algorithm_first"),
        "heuristic_algorithm": ("template_expansion", "algorithm_first"),
        "repair_ablation": ("repair_or_ablation", "mixed"),
        "heuristic_tuning": ("parameter_sanity", "conservative"),
        "novelty_spec": ("high_risk_spec", "explore"),
    }
    action_defaults = {
        "mechanism_discovery": algorithm_stage.get("preferred_actions") or explore_stage.get("preferred_actions") or [],
        "template_expansion": algorithm_stage.get("preferred_actions") or [],
        "repair_or_ablation": repair_stage.get("preferred_actions") or [],
        "parameter_sanity": exploit_stage.get("preferred_actions") or ["parameter_tuning"],
        "high_risk_spec": explore_stage.get("preferred_actions") or [],
    }
    for producer_id, target_count in allocation.items():
        if target_count <= 0:
            continue
        role, proposal_mode = role_map[producer_id]
        directives.append(
            ProducerDirective(
                producer_id=producer_id,
                role=role,
                proposal_mode=proposal_mode,
                target_count=target_count,
                priority_families=priority_families,
                preferred_actions=[str(item) for item in action_defaults.get(role, []) if str(item)],
                avoid_families=sorted(set(avoid_families)),
                instruction=_producer_instruction(role),
            )
        )
    return directives


def _producer_instruction(role: str) -> str:
    if role == "mechanism_discovery":
        return "Generate algorithmic mechanisms with explicit novelty, ablation parent, and failure mode."
    if role == "template_expansion":
        return "Expand from high-potential wired parents using declared local extension mechanisms."
    if role == "repair_or_ablation":
        return "Repair a promising family with one-axis changes or produce a clean ablation parent."
    if role == "parameter_sanity":
        return "Use parameter-only proposals only as sanity checks around runnable parents."
    return "Produce a higher-risk research spec without changing protocol or RecBole core."


def producer_directives_payload(directives: list[ProducerDirective]) -> list[dict[str, Any]]:
    return [
        {
            "producer_id": item.producer_id,
            "role": item.role,
            "proposal_mode": item.proposal_mode,
            "target_count": item.target_count,
            "priority_families": item.priority_families,
            "preferred_actions": item.preferred_actions,
            "avoid_families": item.avoid_families,
            "instruction": item.instruction,
        }
        for item in directives
    ]


def select_producer_for_proposal(
    proposal: dict[str, Any],
    directives: list[ProducerDirective],
    *,
    default_source: str = "",
) -> ProducerDirective | None:
    if not directives:
        return None
    explicit = str(proposal.get("producer_id") or "")
    if explicit:
        for directive in directives:
            if directive.producer_id == explicit:
                return directive
    proposal_type = str(proposal.get("proposal_type") or "")
    action = str(proposal.get("action_type") or "")
    runnable = str(proposal.get("runnable_level") or "")
    if proposal_type == "tuning" or action == "parameter_tuning":
        target_role = "parameter_sanity"
    elif runnable == "spec_only":
        target_role = "high_risk_spec"
    elif "heuristic" in default_source:
        target_role = "template_expansion"
    elif action in {"regularization", "aggregation"}:
        target_role = "repair_or_ablation"
    else:
        target_role = "mechanism_discovery"
    for directive in directives:
        if directive.role == target_role:
            return directive
    return directives[0]


def annotate_proposals(
    proposals: list[dict[str, Any]],
    *,
    directives: list[ProducerDirective],
    default_source: str = "",
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for proposal in proposals:
        item = dict(proposal)
        directive = select_producer_for_proposal(item, directives, default_source=default_source)
        if directive is not None:
            item.setdefault("producer_id", directive.producer_id)
            item.setdefault("producer_role", directive.role)
            item.setdefault("research_line", {})
            if isinstance(item["research_line"], dict):
                item["research_line"].update(
                    {
                        "producer_id": directive.producer_id,
                        "producer_role": directive.role,
                        "producer_instruction": directive.instruction,
                    }
                )
        annotated.append(item)
    return annotated


def _risk_penalty(proposal: dict[str, Any]) -> float:
    risk = proposal.get("risk")
    text = json.dumps(risk, ensure_ascii=False).lower() if isinstance(risk, dict) else str(risk or "").lower()
    penalty = 0.0
    if "recbole_core_change_required" in text and "true" in text:
        penalty += 1.0
    if "protocol" in text and any(word in text for word in ("change", "modify", "alter")):
        penalty += 0.35
    if any(word in text for word in RUNTIME_RISK_KEYWORDS):
        penalty += 0.08
    return penalty


def route_proposal(
    proposal: dict[str, Any],
    *,
    search_memory: dict[str, Any] | None = None,
    search_policy: dict[str, Any] | None = None,
    validation_status: dict[str, Any] | None = None,
) -> RouteDecision:
    memory = search_memory or {}
    policy = search_policy or {}
    candidate_id = str(proposal.get("candidate_id") or "")
    family = str(proposal.get("parent_candidate_id") or candidate_id)
    action = str(proposal.get("action_type") or "")
    proposal_type = str(proposal.get("proposal_type") or "")
    complexity = str(proposal.get("implementation_complexity") or "medium").lower()
    tokens = set(mechanism_tokens(proposal))
    seen = set(memory.get("seen_signatures") or [])
    signature = normalize_signature(proposal.get("parameter_signature"))
    family_stats = memory.get("family_stats") if isinstance(memory.get("family_stats"), dict) else {}
    family_row = family_stats.get(family) if isinstance(family_stats.get(family), dict) else {}
    frozen = [str(item) for item in memory.get("frozen_families", [])]
    promising = [str(item) for item in memory.get("promising_families", [])]
    policy_algorithm = policy.get("algorithm_first") if isinstance(policy.get("algorithm_first"), dict) else {}
    anchors = [str(item) for item in policy_algorithm.get("anchor_families", []) if str(item)]

    duplicate_penalty = 0.0
    if signature and signature in seen:
        duplicate_penalty += 1.25
    duplicate_penalty += max(0, int(family_row.get("runs") or 0) - 3) * 0.025
    frozen_penalty = 0.0
    if any(family_matches(family, item) for item in frozen):
        frozen_penalty += 0.55
    blocker_penalty = min(0.35, 0.08 * int(family_row.get("crashes") or 0))
    novelty = min(0.42, 0.10 + 0.07 * len(tokens))
    information = ACTION_INFORMATION_PRIOR.get(action, 0.05)
    if proposal_type == "algorithmic_variant":
        information += 0.12
    if str(proposal.get("runnable_level") or "") == "code_required":
        information += 0.04
    parent_credit = 0.0
    if any(family_matches(family, item) for item in anchors):
        parent_credit += 0.18
    if any(family_matches(family, item) for item in promising):
        parent_credit += 0.12
    if int(family_row.get("keeps") or 0) > 0:
        parent_credit += 0.08
    executable_credit = 0.0
    if str(proposal.get("runnable_level") or "") in {"parameter_only", "config_only"}:
        executable_credit += 0.10
    if proposal.get("ablation_parent"):
        executable_credit += 0.07
    validation_penalty = 0.0
    validation_name = str((validation_status or {}).get("status") or "")
    if validation_name == "rejected":
        validation_penalty += 2.0
    elif validation_name == "needs_review":
        validation_penalty += 0.08
    elif validation_name == "accepted":
        executable_credit += 0.12
    risk = _risk_penalty(proposal)
    complexity_penalty = COMPLEXITY_COST.get(complexity, 0.08)
    factors = {
        "novelty": novelty,
        "information": information,
        "parent_credit": parent_credit,
        "executable_credit": executable_credit,
        "duplicate_penalty": -duplicate_penalty,
        "blocker_penalty": -blocker_penalty,
        "frozen_penalty": -frozen_penalty,
        "risk_penalty": -risk,
        "complexity_penalty": -complexity_penalty,
        "validation_penalty": -validation_penalty,
    }
    score = sum(factors.values())
    if validation_name == "rejected" or risk >= 1.0:
        decision = "reject"
    elif score < -0.25:
        decision = "downrank"
    else:
        decision = "route"
    reason = (
        f"family={family}; action={action or 'unknown'}; "
        f"producer={proposal.get('producer_id') or proposal.get('producer_role') or 'unknown'}"
    )
    return RouteDecision(candidate_id=candidate_id, score=score, factors=factors, decision=decision, reason=reason)


def route_proposals(
    proposals: list[dict[str, Any]],
    *,
    search_memory: dict[str, Any] | None = None,
    search_policy: dict[str, Any] | None = None,
    validation_results: list[dict[str, Any]] | None = None,
) -> list[RouteDecision]:
    statuses = {
        str(item.get("candidate_id") or ""): item
        for item in (validation_results or [])
        if isinstance(item, dict)
    }
    decisions = [
        route_proposal(
            proposal,
            search_memory=search_memory,
            search_policy=search_policy,
            validation_status=statuses.get(str(proposal.get("candidate_id") or "")),
        )
        for proposal in proposals
    ]
    return sorted(decisions, key=lambda item: item.score, reverse=True)


def order_proposals_by_route(
    proposals: list[dict[str, Any]],
    *,
    search_memory: dict[str, Any] | None = None,
    search_policy: dict[str, Any] | None = None,
    validation_results: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    by_id = {str(item.get("candidate_id") or ""): dict(item) for item in proposals}
    ordered: list[dict[str, Any]] = []
    for decision in route_proposals(
        proposals,
        search_memory=search_memory,
        search_policy=search_policy,
        validation_results=validation_results,
    ):
        item = by_id.get(decision.candidate_id)
        if item is None:
            continue
        item.setdefault("research_line", {})
        if isinstance(item["research_line"], dict):
            item["research_line"]["router"] = decision.as_dict()
        if decision.decision != "reject":
            ordered.append(item)
        if limit is not None and len(ordered) >= limit:
            break
    return ordered


def route_candidate_option(
    candidate: dict[str, Any],
    *,
    base_score: float,
    search_memory: dict[str, Any] | None = None,
    pending_bonus: float = 0.0,
    algorithm_bonus: float = 0.0,
) -> RouteDecision:
    memory = search_memory or {}
    family = str(candidate.get("parent_candidate_id") or candidate.get("candidate_id") or "")
    family_stats = memory.get("family_stats") if isinstance(memory.get("family_stats"), dict) else {}
    row = family_stats.get(family) if isinstance(family_stats.get(family), dict) else {}
    action = str(candidate.get("action_type") or "")
    information = ACTION_INFORMATION_PRIOR.get(action, 0.06)
    pending = max(0.0, pending_bonus)
    algorithm = algorithm_bonus
    blocker = min(0.30, 0.08 * int(row.get("crashes") or 0))
    repeat = min(0.25, 0.03 * int(row.get("runs") or 0))
    factors = {
        "base_score": base_score,
        "information": information,
        "pending_bonus": pending,
        "algorithm_bonus": algorithm,
        "blocker_penalty": -blocker,
        "repeat_penalty": -repeat,
    }
    score = sum(factors.values())
    return RouteDecision(
        candidate_id=str(candidate.get("candidate_id") or ""),
        score=score,
        factors=factors,
        decision="route",
        reason=f"family={family}; action={action or 'unknown'}",
    )


def meta_research_update(search_memory: dict[str, Any]) -> dict[str, Any]:
    producer_rows = list((search_memory.get("producer_stats") or {}).values())
    producer_rows = [row for row in producer_rows if isinstance(row, dict)]
    producer_rows.sort(
        key=lambda row: (
            (int(row.get("keeps") or 0) - int(row.get("crashes") or 0)),
            to_float(row.get("best")) if to_float(row.get("best")) is not None else -1,
        ),
        reverse=True,
    )
    weak_producers = [
        str(row.get("producer_id"))
        for row in producer_rows
        if int(row.get("trials") or 0) >= 3 and int(row.get("keeps") or 0) == 0 and int(row.get("crashes") or 0) > 0
    ]
    return {
        "status": "advisory",
        "champion_producers": [str(row.get("producer_id")) for row in producer_rows[:3] if row.get("producer_id")],
        "downweight_producers": weak_producers[:3],
        "memory_transformations": [
            "deduplicate by parent parameter_signature and execution_signature",
            "track blocker signatures separately from algorithm mechanism failures",
            "promote families only after same-protocol multi-seed evidence",
        ],
        "router_updates": [
            "increase information-gain weight for underexplored algorithmic actions",
            "penalize repeated local repair after plateau detection",
            "route parameter-only tuning after a credible algorithm signal",
        ],
    }
