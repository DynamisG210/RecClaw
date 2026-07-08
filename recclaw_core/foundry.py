"""Golden fixture replay CLI for RecClaw Candidate Foundry v0.1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from recclaw_core.candidate import validate_candidate_cards
from recclaw_core.contracts import NATIVE_BPR_RANKCUT_MEMORY_ID, dump_yaml, load_yaml
from recclaw_core.memory import ingest_bpr_rankcut_memory, retrieve_memory
from recclaw_core.policy import rank_candidates, select_candidate_queue
from recclaw_core.verifier import verify_claim_boundary


def _load_candidate_cards(fixture: Path) -> List[Dict[str, Any]]:
    payload = load_yaml(fixture / "candidate_cards.yaml")
    cards = payload.get("candidate_cards", [])
    if not isinstance(cards, list):
        raise ValueError("candidate_cards.yaml must contain candidate_cards list")
    return [row for row in cards if isinstance(row, dict)]


def replay_fixture(fixture: Path, output: Path) -> Dict[str, Any]:
    memory = ingest_bpr_rankcut_memory(fixture / "phase9_5")
    cards = _load_candidate_cards(fixture)
    validation = validate_candidate_cards(cards)
    if validation["status"] != "candidate_validation_passed":
        output.mkdir(parents=True, exist_ok=True)
        dump_yaml(output / "candidate_validation_report.yaml", validation)
        return {"status": "candidate_validation_failed", "output": output.as_posix()}

    retrieval = {
        "status": "memory_retrieval_checked",
        "query": "future BPR rank surrogate candidate using batch local cutoff",
        "retrieved_memories": retrieve_memory(
            "future BPR rank surrogate candidate using batch local cutoff",
            [memory],
        ),
    }
    ranking_without_memory = rank_candidates(cards, [])
    ranking_with_memory = rank_candidates(cards, [memory])
    queue = select_candidate_queue(ranking_with_memory["ranked_candidates"])
    top_without = ranking_without_memory["ranked_candidates"][0]["candidate_id"]
    top_with = ranking_with_memory["ranked_candidates"][0]["candidate_id"]
    claim_report = {
        "status": "candidate_foundry_fixture_replayed",
        "supported_scope": "candidate_foundry_fixture_replay_only",
        "typed_memory_ingested": memory["memory_id"] == NATIVE_BPR_RANKCUT_MEMORY_ID,
        "retrieval_checked": bool(retrieval["retrieved_memories"]),
        "deterministic_policy_scored": True,
        "selected_candidate_queue_written": True,
        "memory_changes_top_candidate": top_without != top_with,
        "top_candidate_without_memory": top_without,
        "top_candidate_with_memory": top_with,
        "search_quality_supported": False,
        "metric_improvement_supported": False,
        "formal_success_supported": False,
        "M5_ready": False,
        "runtime_kernel_ready": False,
    }
    claim_verification = verify_claim_boundary(claim_report)

    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(output / "typed_memory_card.yaml", memory)
    dump_yaml(output / "memory_retrieval_report.yaml", retrieval)
    dump_yaml(output / "candidate_validation_report.yaml", validation)
    dump_yaml(output / "policy_scores_without_memory.yaml", ranking_without_memory)
    dump_yaml(output / "policy_scores_with_memory.yaml", ranking_with_memory)
    dump_yaml(output / "selected_candidate_queue.yaml", queue)
    dump_yaml(output / "claim_boundary_report.yaml", claim_report)
    dump_yaml(output / "claim_boundary_verification.yaml", claim_verification)
    return {
        "status": "candidate_foundry_fixture_replayed",
        "output": output.as_posix(),
        "top_candidate_without_memory": top_without,
        "top_candidate_with_memory": top_with,
        "claim_boundary_status": claim_verification["status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a RecClaw Candidate Foundry golden fixture.")
    parser.add_argument("--fixture", required=True, type=Path, help="Golden fixture directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for replay artifacts")
    args = parser.parse_args(argv)
    result = replay_fixture(args.fixture, args.output)
    for key, value in result.items():
        print(f"{key}={value}")
    return 0 if result.get("claim_boundary_status") in {None, "claim_boundary_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
