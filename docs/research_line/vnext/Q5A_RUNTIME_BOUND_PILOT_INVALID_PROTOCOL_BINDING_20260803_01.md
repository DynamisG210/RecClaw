# Q5-A Runtime-Bound Pilot — Invalid Protocol Binding

Status: HARD_BLOCK_INVALID_PROTOCOL_BINDING.

The fresh campaign completed the frozen 3×8 Provider pool: 24 physical Provider calls, zero retry/replacement/filtering, and the complete attempt ledger is preserved. The later SELECT path exposed a non-atomic write boundary and a missing Q5A-to-Q3 envelope field. A corrected SELECT then produced 16 unique specs, but it was produced by a runner whose hash was not the prefreeze/deployment-bound hash. Those artifacts remain sealed evidence only and are not valid campaign inputs.

The mandatory REALIZE gate first rejected the patched runner with runtime dependency hash drift: runner_entrypoint. An explicit stop then terminated the leaked remote process tree. Partial realization directories and stage traces are preserved for audit; they yielded 26 preflight/resolver invocations, 13 resolver failures, zero implementer/qualifier/admission/training/mechanism/effect receipts, and zero new physical attempt directories. The stale EXECUTION_OWNER.json is retained as evidence and is not cleared.

No selection, realization, effect, mechanism-information, or scientific authority is consumable from this campaign. Do not run Q5-B or restart this root.

The generic repair in this branch makes SELECT transaction-bound: it validates the accepted policy/activation pair and Q5A-to-Q3 adapter before publishing any selection artifact, builds all 3×3 selections and the union in memory, and publishes only after the full computation succeeds. Targeted tests cover pair mismatch, adapter mismatch, and successful offline 3×8-shaped selection with zero partial artifacts.

Evidence: results/research_line/q5a_runtime_bound_pilot_20260803_01/Q5A_HARD_BLOCK_INVALID_PROTOCOL_BINDING.json.
