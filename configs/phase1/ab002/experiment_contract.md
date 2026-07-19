# AB-002 Clean-Start Matched Development A/B Contract

The machine authority is `experiment_contract.json`; this file is a concise human-readable projection. Status is `LOCAL_COMPLETE`, gate status is `NOT_STARTED`, authority is `NONE`, and all outputs are `DEVELOPMENT_ONLY`.

Both arms start from `AB002-RECONSTRUCTED-CLEAN-START-S0-V1` with source-tree digest `47f478ca51f2ea3703861ab288513462f6fdc7b933901814303845e6acd296cf`. The reconstruction uses the pre-search code/config registry at Git commit `b3dc5a7e33fd1cbec60b5dfb9ae7097c67d0b5e8`, current common runtime launch components, an empty search memory, and no historical proposals, candidates, search trees, repair memory, or best scores.

The fixed protocol is ML-1M, RecBole 1.2.1, random per-user holdout 0.8/0.1/0.1, uniform one-negative training sampling, full-sort validation/test, and NDCG@10 as the primary metric. The exact dataset tree digest is `532f0c05827ee06d1b5f81de9686bb8cf4288e67260a55853bc45d092c82f9dd` under the algorithm recorded in the machine contract.

The Canary uses paired search seed 9001, three rounds per arm, at most six candidate executions per arm, and one implementation attempt per round. It is excluded from outcomes. Full A/B uses paired seeds 42–47, 20 rounds and at most 20 candidate executions per arm. Ordinary training uses seed 2026; the fixed LightGCN comparator and selected new candidate use seeds 2026, 2027, and 2028.

The Agent model is exactly `gpt-5.4`, temperature 0.2, timeout 300 seconds, retries 3, with reasoning effort and service tier omitted as in Original RecClaw. A common broker replays exact response bytes only for identical canonical requests within the same pair and occurrence; divergent prompts receive separate calls. It stores request/response provenance but no credentials.

Control contains no Guard source, import, state, output, or feedback. Treatment adds only the frozen overlay and frozen development Evidence Guard files. Execution permission, evidence admission, claim authority, artifact promotion, and Canary/Full gates remain separate.

The frozen arm-blind Neutral Outcome Auditor checks raw logs, result JSON, candidate artifacts, dataset bytes, RecBole critical-file bytes, split, sampling, full-sort mode, seeds, and metrics. Full-run candidate selection is performed only on blinded packets that exclude `agent.py`, Guard feedback, and arm labels. No new eligible candidate means the fixed LightGCN comparator is the frontier.

Canary completion may only produce a package ready for independent read-only review. It does not authorize the six-pair Full A/B. Full execution requires an external Canary GO and the separate mechanical environment marker enforced by the runner.
