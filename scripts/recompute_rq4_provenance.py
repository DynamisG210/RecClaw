"""Reconcile original RQ4 observations, later branches and paper claims.

Read-only inputs; no model requests or training. All paths are explicit CLI inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def comparison(rows):
    paired = []
    for batch, state in dict.fromkeys((r["batch"], r["state"]) for r in rows):
        arms = {r["feedback"]: r["ndcg5"] for r in rows
                if (r["batch"], r["state"]) == (batch, state)}
        if arms.get("Mechanism") is not None and arms.get("Summary") is not None:
            paired.append({"batch": batch, "state": state,
                           "delta_mechanism_minus_summary": arms["Mechanism"] - arms["Summary"]})
    deltas = [r["delta_mechanism_minus_summary"] for r in paired]
    return {"paired_states": len(deltas), "wins": sum(d > 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
            "mean_signed_ndcg5_delta": mean(deltas), "pairs": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.original_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["state"] = int(row["state"])
        row["ndcg5"] = float(row["ndcg5"]) if row["ndcg5"] else None
    keys = [(r["batch"], r["state"], r["feedback"]) for r in rows]
    assert len(rows) == len(set(keys)) == 21, "Expected original 21 unique cells"
    order = list(dict.fromkeys((r["batch"], r["state"]) for r in rows))
    assert [s for _, s in order] == [2, 3, 18, 6, 10, 8, 12]
    result_path = args.fresh_dir / "delivery/NEW_MECHANISM_RESULT.json"
    fresh = read_json(result_path)
    protocol_path = args.fresh_dir / "FRESH_BRANCH_PROTOCOL.json"
    protocol = read_json(protocol_path)
    old = next(r for r in rows if (r["batch"], r["state"], r["feedback"]) ==
               ("Source-grounded", 6, "Mechanism"))
    assert fresh["original_mechanism_ndcg5"] == old["ndcg5"]
    assert fresh["state"] == protocol["start_state"] == 6
    assert fresh["common_raw_sha256"] == protocol["common_raw_sha256"]
    assert fresh["new_physical_fit"] and not fresh["test_accessed"]
    assert fresh["fresh_steps"]["full_fits"] == 1
    inventory = [args.original_csv, result_path, protocol_path]
    native_sources = []
    for name in dict.fromkeys(r["source"] for r in rows):
        path = Path(name)
        native_sources.append({"path": str(path), "exists": path.is_file(),
                               "sha256": digest(path) if path.is_file() else None})
        if path.is_file():
            inventory.append(path)
    original_native = read_json(Path(old["source"]))
    original_cell = next(r for r in original_native["rows"]
                         if r["state"] == 6 and r["arm"] == "Mechanism")
    assert original_cell["metric"] == old["ndcg5"]
    candidate_checks = []
    for name, expected in fresh["files_sha256"].items():
        path = args.fresh_dir / "delivery/candidate" / name
        actual = digest(path)
        assert actual == expected, f"Candidate source mismatch: {name}"
        candidate_checks.append({"file": name, "sha256": actual, "match": True})
        inventory.append(path)
    workers = list((args.fresh_dir / "delivery/evidence").rglob("worker_result.json"))
    matches = [p for p in workers if digest(p) == fresh["worker_result_sha256"]]
    assert len(matches) == 1, "Fresh physical worker receipt must match recorded hash"
    inventory.append(matches[0])
    convergence_path = args.fresh_dir / "FEEDBACK_CONVERGENCE_AUDIT.json"
    convergence = read_json(convergence_path)
    inventory.append(convergence_path)
    decision_trace = {"identical_mechanism_feedback_request":
                      convergence["old_new_mechanism_request_exact_equal"],
                      "interpretation": "Different outputs from identical requests show within-arm proposal variation, not a feedback-treatment causal contrast."}
    for name in ("Mechanism_original", "Mechanism_fresh", "Mechanism_old", "Summary", "Raw"):
        if name in convergence:
            entry = convergence[name]
            decision_trace[name] = {"proposals": entry.get("proposals"),
                                    "feedback_request_sha256": entry.get("feedback_request_sha256"),
                                    "common_facts_sha256": entry.get("common_facts_sha256")}
    paper_table = args.paper_dir / "tables/RQ4_all_seven_states.tex"
    correction = args.paper_dir / "docs/RQ3_SINGLE_PANEL_AND_RQ4_CORRECTION_20261022.md"
    experiments = args.paper_dir / "sections/04_experiments.tex"
    inventory.extend([paper_table, correction, experiments])
    table = paper_table.read_text(encoding="utf-8")
    disclosure = correction.read_text(encoding="utf-8")
    paper_claim = experiments.read_text(encoding="utf-8")
    counterfactual = [dict(r) for r in rows]
    for r in counterfactual:
        if (r["batch"], r["state"], r["feedback"]) == ("Source-grounded", 6, "Mechanism"):
            r["ndcg5"] = fresh["metrics"]["ndcg@5"]
    output = {
        "schema": "recrsi.rq4.observation-provenance.v1",
        "evidence_class": "offline recomputation, no new measurement",
        "new_calls": 0, "new_fits": 0, "new_cost_usd": 0,
        "original_cells": rows,
        "paper_row_mapping": [{"paper_row": i + 1, "batch": b, "original_state": s}
                              for i, (b, s) in enumerate(order)],
        "original_mechanism_vs_summary": comparison(rows),
        "original_missing_numeric_cells": [r for r in rows if r["ndcg5"] is None],
        "later_observation": {"paper_row": 4, "original_state": 6,
            "original_ndcg5": old["ndcg5"], "fresh_ndcg5": fresh["metrics"]["ndcg@5"],
            "original_action": old["action"], "fresh_mechanism": fresh["mechanism"],
            "original_native_cell": original_cell,
            "fresh_result_path": str(result_path), "fresh_result": fresh},
        "replacement_sensitivity_only_not_primary_estimate": comparison(counterfactual),
        "paper_disclosure_checks": {
            "table_contains_original_negative": "0.3656" in table,
            "correction_says_not_new_experiment": "不视为新增实验" in disclosure,
            "paper_claims_five_leads_two_ties": "five strict leads and two ties" in paper_claim,
            "finding": "Fresh feedback, proposals, implementation and FIT are a later observation; retain original scientific negative separately."},
        "verified_fresh_candidate_files": candidate_checks,
        "verified_fresh_worker": str(matches[0]), "native_sources": native_sources,
        "decision_trace": decision_trace,
        "limitations": ["Seven states span three development batches; representative sampling not established here.",
                        "No independent search-seed replication or final TEST established.",
                        "Average paired delta is descriptive, not a causal population estimate.",
                        "Original Raw state6 had an engineering failure; never impute zero quality.",
                        "Full historical cost total not reconstructed; fresh reference cost is not provider cash."],
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in dict.fromkeys(inventory)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "original": output["original_mechanism_vs_summary"],
                      "replacement_sensitivity": output["replacement_sensitivity_only_not_primary_estimate"],
                      "hash_verified_candidate_files": len(candidate_checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
