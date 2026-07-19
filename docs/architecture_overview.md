# Architecture overview

Phase 1 asks one bounded empirical question: does adding the trustworthy-evidence line to the final effective Original RecClaw change the research trajectory under a matched development protocol?

```text
                    common paired LLM broker
                       /                \
Original RecClaw Control          Original RecClaw Treatment
producer -> selector -> runner    producer -> selector -> runner
         -> raw memory                     -> Evidence Guard
                                            -> admitted primary memory
                                            -> diagnostic/quarantine memory
                       \                /
                    AB-002 analysis
                           |
                 Neutral Outcome Auditor
```

The causal comparison changes only the Treatment feedback boundary. Dataset, protocol, candidate producer, selector, runner, initial empty S0, baseline, search seeds, training seed policy, budgets, provider/model parameters, and contemporaneous scheduling are held fixed.

The common broker is not a research policy. It assigns a per-arm sequence within each pair, makes one upstream call for canonical-identical paired requests, replays the exact response to the other arm, and records divergent prompts instead of pretending they were matched.

Control materialization copies only the Original RecClaw runtime allowlist. It omits `src/recclaw_core`, Guard contracts, Treatment overlays, Guard environment variables, and every file whose name or content imports the Guard. Treatment materialization copies the same Original source, applies the frozen overlay, and supplies only the five hash-bound integration files plus the fixed contract.

All training, broker, log, model, temporary source, and analysis outputs live under an explicit external runtime root. The Git checkout is an immutable source input during dry-run and execution.
