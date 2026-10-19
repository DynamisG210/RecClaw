"""Complete-pool held-out evaluation and immutable Meta VNext decisions."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from ..canonical import sha256_digest, validate_sha256
from .contracts import (
    CompletePoolEpisodeV1,
    EpisodeSplitV1,
    HeldoutFreshnessV1,
    MetaVNextRecord,
    SearchValueObservationV1,
)
from .learning import PairwiseSlowPolicyV1
from .routing import (
    MetaVNextRouterV1,
    initialize_fast_residual,
    static_champion_candidate,
    update_fast_residual,
)


class MetaVNextEvaluationError(ValueError):
    """Raised when policies are not evaluated on the same complete support."""


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _group_bootstrap_interval(
    group_differences: Sequence[float],
    *,
    confidence_level: float,
    seed_digest: str,
    sample_count: int = 10000,
) -> tuple[float, float]:
    count = len(group_differences)
    if count <= 7:
        samples = [
            _mean([group_differences[index] for index in indices])
            for indices in itertools.product(range(count), repeat=count)
        ]
    else:
        seed = int(seed_digest[:16], 16)
        rng = random.Random(seed)
        samples = [
            _mean(
                [
                    group_differences[rng.randrange(count)]
                    for _ in range(count)
                ]
            )
            for _ in range(sample_count)
        ]
    tail = (1.0 - confidence_level) / 2.0
    return (
        round(_quantile(samples, tail), 12),
        round(_quantile(samples, 1.0 - tail), 12),
    )


@dataclass(frozen=True, slots=True)
class PolicyEpisodeObservationV1(MetaVNextRecord):
    policy_name: str
    episode_id: str
    episode_group_id: str
    selected_candidate_semantics_digest: str
    producer_id: str
    mechanism_family_id: str
    mechanism_axis: str
    change_class: str
    net_research_value: float
    oracle_net_research_value: float
    regret: float
    frontier_value: float
    discriminative_value: float
    normalized_cost: float
    blocker_loss: float

    schema = "recclaw.meta-vnext.policy-episode-observation.v1"


@dataclass(frozen=True, slots=True)
class PolicyAggregateV1(MetaVNextRecord):
    policy_name: str
    mean_net_research_value: float
    mean_regret: float
    useful_signal_rate: float
    mean_frontier_value: float
    mean_discriminative_value: float
    mean_normalized_cost: float
    mean_blocker_loss: float
    producer_max_share: float
    family_max_share: float
    axis_max_share: float
    tuning_share: float
    all_candidate_value_mae: float | None
    all_candidate_value_rmse: float | None
    observations: tuple[PolicyEpisodeObservationV1, ...]

    schema = "recclaw.meta-vnext.policy-aggregate.v1"


@dataclass(frozen=True, slots=True)
class PairedComparisonV1(MetaVNextRecord):
    challenger: str
    incumbent: str
    mean_net_value_improvement: float
    mean_regret_improvement: float
    group_bootstrap_net_improvement_interval: tuple[float, float]
    mean_cost_difference: float
    mean_blocker_difference: float
    per_group_net_value_improvement: tuple[tuple[str, float], ...]

    schema = "recclaw.meta-vnext.paired-comparison.v1"


@dataclass(frozen=True, slots=True)
class PromotionCriteriaV1(MetaVNextRecord):
    minimum_group_count: int
    confidence_level: float
    minimum_slow_net_improvement: float
    minimum_fast_increment: float
    cost_noninferiority_margin: float
    blocker_noninferiority_margin: float

    schema = "recclaw.meta-vnext.promotion-criteria.v1"

    def __post_init__(self) -> None:
        if self.minimum_group_count < 2:
            raise MetaVNextEvaluationError("minimum_group_count must be at least two")
        if not 0.0 < self.confidence_level < 1.0:
            raise MetaVNextEvaluationError("confidence_level must be in (0,1)")
        if self.minimum_slow_net_improvement <= 0.0:
            raise MetaVNextEvaluationError(
                "slow promotion requires a positive minimum effect"
            )
        if self.minimum_fast_increment <= 0.0:
            raise MetaVNextEvaluationError(
                "Meta promotion requires a positive fast increment"
            )
        if self.cost_noninferiority_margin < 0.0:
            raise MetaVNextEvaluationError("cost margin cannot be negative")
        if self.blocker_noninferiority_margin < 0.0:
            raise MetaVNextEvaluationError("blocker margin cannot be negative")


@dataclass(frozen=True, slots=True)
class MetaVNextEvaluationReportV1(MetaVNextRecord):
    evaluation_id: str
    policy_digest: str
    slow_policy_digest: str
    router_configuration_digest: str
    heldout_episode_set_digest: str
    heldout_freshness: HeldoutFreshnessV1
    independent_group_count: int
    complete_pool_support: bool
    same_episode_set: bool
    candidate_id_leakage_detected: bool
    deterministic_replay: bool
    static_champion: PolicyAggregateV1
    slow_ranker: PolicyAggregateV1
    slow_plus_fast: PolicyAggregateV1
    slow_vs_static: PairedComparisonV1
    fast_vs_slow: PairedComparisonV1
    learned_router_recommendation: str
    meta_promotion_recommendation: str
    criteria: PromotionCriteriaV1

    schema = "recclaw.meta-vnext.evaluation-report.v1"

    def __post_init__(self) -> None:
        validate_sha256(self.policy_digest, field_name="policy_digest")
        validate_sha256(
            self.slow_policy_digest,
            field_name="slow_policy_digest",
        )
        validate_sha256(
            self.router_configuration_digest,
            field_name="router_configuration_digest",
        )
        validate_sha256(
            self.heldout_episode_set_digest,
            field_name="heldout_episode_set_digest",
        )
        if self.learned_router_recommendation not in {
            "PROMOTE",
            "HOLD",
            "INCONCLUSIVE",
        }:
            raise MetaVNextEvaluationError("invalid learned-router recommendation")
        if self.meta_promotion_recommendation not in {
            "PROMOTE",
            "HOLD",
            "INCONCLUSIVE",
        }:
            raise MetaVNextEvaluationError("invalid Meta recommendation")


@dataclass(frozen=True, slots=True)
class MetaVNextPromotionDecisionV1(MetaVNextRecord):
    decision_id: str
    verdict: str
    claim_class: str
    policy_digest: str
    parent_policy_digest: str
    evaluation_report_digest: str
    heldout_episode_set_digest: str
    activation_boundary: str
    rollback_parent_digest: str

    schema = "recclaw.meta-vnext.promotion-decision.v1"

    def __post_init__(self) -> None:
        if self.verdict != "PROMOTE" or self.claim_class != "META_LEARNING":
            raise MetaVNextEvaluationError(
                "immutable Meta decision is created only for a passed Meta gate"
            )
        if self.activation_boundary not in {
            "NEXT_ROUND",
            "NEXT_CAMPAIGN",
        }:
            raise MetaVNextEvaluationError("invalid activation boundary")
        for name in (
            "policy_digest",
            "parent_policy_digest",
            "evaluation_report_digest",
            "heldout_episode_set_digest",
            "rollback_parent_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)


def _selected_observation(
    *,
    policy_name: str,
    episode: CompletePoolEpisodeV1,
    selected_digest: str,
    policy: PairwiseSlowPolicyV1,
) -> PolicyEpisodeObservationV1:
    candidates = {
        item.candidate_semantics_digest: item
        for item in episode.pool.eligible_candidates
    }
    outcomes = episode.outcomes_by_semantics
    selected = outcomes[selected_digest]
    values = {
        digest: outcome.net_research_value(policy.value_weights)
        for digest, outcome in outcomes.items()
    }
    selected_candidate = candidates[selected_digest]
    selected_value = values[selected_digest]
    oracle = max(values.values())
    return PolicyEpisodeObservationV1(
        policy_name=policy_name,
        episode_id=episode.episode_id,
        episode_group_id=episode.episode_group_id,
        selected_candidate_semantics_digest=selected_digest,
        producer_id=selected_candidate.producer_id,
        mechanism_family_id=selected_candidate.mechanism_family_id,
        mechanism_axis=selected_candidate.primary_mechanism_axis,
        change_class=selected_candidate.change_class.value,
        net_research_value=selected_value,
        oracle_net_research_value=oracle,
        regret=oracle - selected_value,
        frontier_value=selected.frontier_value,
        discriminative_value=selected.discriminative_value,
        normalized_cost=selected.normalized_cost,
        blocker_loss=selected.blocker_loss,
    )


def _aggregate(
    policy_name: str,
    observations: Sequence[PolicyEpisodeObservationV1],
    *,
    all_candidate_errors: Sequence[float] | None = None,
) -> PolicyAggregateV1:
    items = tuple(observations)
    errors = (
        tuple(float(item) for item in all_candidate_errors)
        if all_candidate_errors is not None
        else None
    )

    def max_share(values: Sequence[str]) -> float:
        counts = Counter(values)
        return max(counts.values()) / len(values)

    return PolicyAggregateV1(
        policy_name=policy_name,
        mean_net_research_value=_mean(
            [item.net_research_value for item in items]
        ),
        mean_regret=_mean([item.regret for item in items]),
        useful_signal_rate=_mean(
            [
                float(
                    item.frontier_value > 0.0
                    or item.discriminative_value > 0.0
                )
                for item in items
            ]
        ),
        mean_frontier_value=_mean([item.frontier_value for item in items]),
        mean_discriminative_value=_mean(
            [item.discriminative_value for item in items]
        ),
        mean_normalized_cost=_mean([item.normalized_cost for item in items]),
        mean_blocker_loss=_mean([item.blocker_loss for item in items]),
        producer_max_share=max_share([item.producer_id for item in items]),
        family_max_share=max_share([item.mechanism_family_id for item in items]),
        axis_max_share=max_share([item.mechanism_axis for item in items]),
        tuning_share=_mean(
            [
                float(item.change_class == "PARAMETER_TUNING_ONLY")
                for item in items
            ]
        ),
        all_candidate_value_mae=(
            _mean([abs(item) for item in errors]) if errors is not None else None
        ),
        all_candidate_value_rmse=(
            math.sqrt(_mean([item * item for item in errors]))
            if errors is not None
            else None
        ),
        observations=items,
    )


def _paired_comparison(
    challenger: PolicyAggregateV1,
    incumbent: PolicyAggregateV1,
    *,
    confidence_level: float,
    seed_digest: str,
) -> PairedComparisonV1:
    challenger_by_episode = {
        item.episode_id: item for item in challenger.observations
    }
    incumbent_by_episode = {
        item.episode_id: item for item in incumbent.observations
    }
    if set(challenger_by_episode) != set(incumbent_by_episode):
        raise MetaVNextEvaluationError("policies were not replayed on the same episodes")
    net_by_group: dict[str, list[float]] = defaultdict(list)
    regret_by_group: dict[str, list[float]] = defaultdict(list)
    cost_by_group: dict[str, list[float]] = defaultdict(list)
    blocker_by_group: dict[str, list[float]] = defaultdict(list)
    for episode_id in sorted(challenger_by_episode):
        left = challenger_by_episode[episode_id]
        right = incumbent_by_episode[episode_id]
        group_id = left.episode_group_id
        difference = left.net_research_value - right.net_research_value
        net_by_group[group_id].append(difference)
        regret_by_group[group_id].append(right.regret - left.regret)
        cost_by_group[group_id].append(
            left.normalized_cost - right.normalized_cost
        )
        blocker_by_group[group_id].append(
            left.blocker_loss - right.blocker_loss
        )
    per_group = tuple(
        (group_id, _mean(values))
        for group_id, values in sorted(net_by_group.items())
    )
    interval = _group_bootstrap_interval(
        [value for _, value in per_group],
        confidence_level=confidence_level,
        seed_digest=seed_digest,
    )
    return PairedComparisonV1(
        challenger=challenger.policy_name,
        incumbent=incumbent.policy_name,
        mean_net_value_improvement=_mean([value for _, value in per_group]),
        mean_regret_improvement=_mean(
            [_mean(values) for _, values in sorted(regret_by_group.items())]
        ),
        group_bootstrap_net_improvement_interval=interval,
        mean_cost_difference=_mean(
            [_mean(values) for _, values in sorted(cost_by_group.items())]
        ),
        mean_blocker_difference=_mean(
            [_mean(values) for _, values in sorted(blocker_by_group.items())]
        ),
        per_group_net_value_improvement=per_group,
    )


def evaluate_heldout_sequences(
    *,
    episodes: Sequence[CompletePoolEpisodeV1],
    policy: PairwiseSlowPolicyV1,
    router: MetaVNextRouterV1,
    criteria: PromotionCriteriaV1,
    evaluation_id: str,
) -> MetaVNextEvaluationReportV1:
    heldout = tuple(episodes)
    if not heldout or any(
        item.split is not EpisodeSplitV1.PROMOTION_HELDOUT for item in heldout
    ):
        raise MetaVNextEvaluationError(
            "promotion evaluation accepts held-out episodes only"
        )
    freshness_values = {item.heldout_freshness for item in heldout}
    if len(freshness_values) != 1:
        raise MetaVNextEvaluationError("held-out freshness must be uniform")
    freshness = next(iter(freshness_values))
    heldout_groups = {episode.episode_group_id for episode in heldout}
    heldout_lineages = {episode.lineage_group_id for episode in heldout}
    if heldout_groups & set(policy.development_group_ids):
        raise MetaVNextEvaluationError(
            "held-out task group overlaps development data"
        )
    if heldout_lineages & set(policy.development_lineage_group_ids):
        raise MetaVNextEvaluationError(
            "held-out task lineage overlaps development data"
        )
    groups: dict[str, list[CompletePoolEpisodeV1]] = defaultdict(list)
    for episode in heldout:
        groups[episode.episode_group_id].append(episode)
    for group_id, items in groups.items():
        ordered = sorted(items, key=lambda item: item.sequence_index)
        if [item.sequence_index for item in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise MetaVNextEvaluationError(
                f"group {group_id} has a non-contiguous task sequence"
            )

    static_observations: list[PolicyEpisodeObservationV1] = []
    slow_observations: list[PolicyEpisodeObservationV1] = []
    fast_observations: list[PolicyEpisodeObservationV1] = []
    slow_value_errors: list[float] = []
    fast_value_errors: list[float] = []
    for group_id in sorted(groups):
        arm_digest = sha256_digest(
            {"evaluation_id": evaluation_id, "arm": "C", "group": group_id}
        )
        seed_digest = sha256_digest(
            {"evaluation_id": evaluation_id, "search_seed_group": group_id}
        )
        fast_state = initialize_fast_residual(
            opaque_arm_instance_digest=arm_digest,
            search_seed_digest=seed_digest,
            policy=policy,
        )
        for episode in sorted(groups[group_id], key=lambda item: item.sequence_index):
            fast_decision = router.route(
                episode.pool, policy, fast_state=fast_state
            )
            fast_score_by_semantics = {
                score.candidate_semantics_digest: score.final_score
                for score in fast_decision.scored_candidates
            }
            for candidate in episode.pool.eligible_candidates:
                observed_value = episode.outcomes_by_semantics[
                    candidate.candidate_semantics_digest
                ].net_research_value(policy.value_weights)
                slow_value_errors.append(policy.score(candidate) - observed_value)
                fast_value_errors.append(
                    fast_score_by_semantics[
                        candidate.candidate_semantics_digest
                    ]
                    - observed_value
                )
            static_candidate = static_champion_candidate(episode.pool)
            static_observations.append(
                _selected_observation(
                    policy_name="STRONG_STATIC_ROUTER_V1",
                    episode=episode,
                    selected_digest=static_candidate.candidate_semantics_digest,
                    policy=policy,
                )
            )
            slow_decision = router.route(episode.pool, policy)
            slow_observations.append(
                _selected_observation(
                    policy_name="META_VNEXT_SLOW",
                    episode=episode,
                    selected_digest=(
                        slow_decision.selected_candidate_semantics_digest
                    ),
                    policy=policy,
                )
            )
            fast_observations.append(
                _selected_observation(
                    policy_name="META_VNEXT_SLOW_PLUS_FAST",
                    episode=episode,
                    selected_digest=(
                        fast_decision.selected_candidate_semantics_digest
                    ),
                    policy=policy,
                )
            )
            selected_candidate = next(
                item
                for item in episode.pool.eligible_candidates
                if item.candidate_semantics_digest
                == fast_decision.selected_candidate_semantics_digest
            )
            selected_outcome: SearchValueObservationV1 = (
                episode.outcomes_by_semantics[
                    fast_decision.selected_candidate_semantics_digest
                ]
            )
            fast_state = update_fast_residual(
                fast_state,
                policy=policy,
                candidate=selected_candidate,
                observation=selected_outcome,
                opaque_arm_instance_digest=arm_digest,
                search_seed_digest=seed_digest,
                round_boundary=episode.sequence_index,
            )

    static = _aggregate("STRONG_STATIC_ROUTER_V1", static_observations)
    slow = _aggregate(
        "META_VNEXT_SLOW",
        slow_observations,
        all_candidate_errors=slow_value_errors,
    )
    fast = _aggregate(
        "META_VNEXT_SLOW_PLUS_FAST",
        fast_observations,
        all_candidate_errors=fast_value_errors,
    )
    heldout_digest = sha256_digest(
        [item.digest for item in sorted(heldout, key=lambda item: item.episode_id)]
    )
    slow_comparison = _paired_comparison(
        slow,
        static,
        confidence_level=criteria.confidence_level,
        seed_digest=sha256_digest(
            {"heldout": heldout_digest, "comparison": "slow-vs-static"}
        ),
    )
    fast_comparison = _paired_comparison(
        fast,
        slow,
        confidence_level=criteria.confidence_level,
        seed_digest=sha256_digest(
            {"heldout": heldout_digest, "comparison": "fast-vs-slow"}
        ),
    )
    insufficient = (
        freshness is not HeldoutFreshnessV1.FRESH_BLINDED
        or len(groups) < criteria.minimum_group_count
    )
    slow_pass = (
        slow_comparison.group_bootstrap_net_improvement_interval[0]
        >= criteria.minimum_slow_net_improvement
        and slow_comparison.mean_cost_difference
        <= criteria.cost_noninferiority_margin
        and slow_comparison.mean_blocker_difference
        <= criteria.blocker_noninferiority_margin
    )
    fast_pass = (
        fast_comparison.group_bootstrap_net_improvement_interval[0]
        >= criteria.minimum_fast_increment
        and fast_comparison.mean_cost_difference
        <= criteria.cost_noninferiority_margin
        and fast_comparison.mean_blocker_difference
        <= criteria.blocker_noninferiority_margin
    )
    if insufficient:
        learned_recommendation = "INCONCLUSIVE"
        meta_recommendation = "INCONCLUSIVE"
    else:
        learned_recommendation = "PROMOTE" if slow_pass else "HOLD"
        meta_recommendation = "PROMOTE" if slow_pass and fast_pass else "HOLD"
    policy_bundle_digest = sha256_digest(
        {
            "schema": "recclaw.meta-vnext.policy-bundle.v1",
            "slow_policy_digest": policy.digest,
            "router_configuration_digest": router.configuration_digest,
        }
    )
    return MetaVNextEvaluationReportV1(
        evaluation_id=evaluation_id,
        policy_digest=policy_bundle_digest,
        slow_policy_digest=policy.digest,
        router_configuration_digest=router.configuration_digest,
        heldout_episode_set_digest=heldout_digest,
        heldout_freshness=freshness,
        independent_group_count=len(groups),
        complete_pool_support=True,
        same_episode_set=True,
        candidate_id_leakage_detected=any(
            "candidate" in name.lower() or "digest" in name.lower()
            for name in policy.feature_names
        ),
        deterministic_replay=True,
        static_champion=static,
        slow_ranker=slow,
        slow_plus_fast=fast,
        slow_vs_static=slow_comparison,
        fast_vs_slow=fast_comparison,
        learned_router_recommendation=learned_recommendation,
        meta_promotion_recommendation=meta_recommendation,
        criteria=criteria,
    )


def create_promotion_decision(
    *,
    report: MetaVNextEvaluationReportV1,
    parent_policy_digest: str,
    activation_boundary: str,
) -> MetaVNextPromotionDecisionV1:
    if (
        report.heldout_freshness is not HeldoutFreshnessV1.FRESH_BLINDED
        or report.meta_promotion_recommendation != "PROMOTE"
    ):
        raise MetaVNextEvaluationError(
            "a recommendation or unblinded report cannot create promotion"
        )
    validate_sha256(parent_policy_digest, field_name="parent_policy_digest")
    return MetaVNextPromotionDecisionV1(
        decision_id=f"meta-vnext-promote-{report.digest[:16]}",
        verdict="PROMOTE",
        claim_class="META_LEARNING",
        policy_digest=report.policy_digest,
        parent_policy_digest=parent_policy_digest,
        evaluation_report_digest=report.digest,
        heldout_episode_set_digest=report.heldout_episode_set_digest,
        activation_boundary=activation_boundary,
        rollback_parent_digest=parent_policy_digest,
    )
