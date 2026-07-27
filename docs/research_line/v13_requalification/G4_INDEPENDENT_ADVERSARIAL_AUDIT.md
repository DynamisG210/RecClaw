# M6G G4 Independent Adversarial Audit

## Scope and method

The audit exercised the live V13 Producer, Router, lineage, task-queue,
mechanism-belief and resource-accounting path. It did not call a Provider,
start training or start a Pilot. Historical V12 identities were treated as
sealed evidence, not as the identity of the superseding V13 source.

## Findings

No P0, P1 or P2 finding remains in G4 scope.

Adversarial checks established:

- all four declared Producers receive discovery credit and
  `falsification_designer` emits `FALSIFICATION` plus a typed
  `DiscriminativeExperimentPlanV1`;
- control and repair are deterministic support services and never receive
  discovery credit;
- declared prior-round parents resolve through the Arm-private
  `LineageIndexV1`; a missing declared parent is rejected instead of falling
  back to a catalog root;
- an explicit root request remains distinct from a missing declared parent;
- every V13 proposal binds a `MatchedControlPlanV1`;
- a missing comparator creates `RUN_MATCHED_CONTROL`, which consumes the next
  normal SearchRound with one execution and zero Broker calls;
- the control result is `DIAGNOSTIC_ONLY`, excluded from the discovery
  frontier and from Meta updates;
- mechanism evidence changes only after an exact same-seed, same-protocol
  primary/control comparison with an explicit changed axis;
- without that comparison, the observation carries no evidence-for or
  evidence-against transition;
- Router features change with compile/handler/materializer facts, exact
  parent availability, semantic duplicates, mechanism depth, cost and typed
  blocker history;
- `utility_floor` rejects a low-utility candidate with
  `UTILITY_BELOW_FLOOR`;
- static Research routing remains active when Meta is absent;
- Broker call latencies, proposal-session wall time, training wall time,
  round total wall time and GPU use are separately present in the sealed
  round result; V13 wall debit includes successful proposal latency.

## Verification

Targeted G4 tests:

```text
8 passed
```

Affected Research, Helix, Original, frontier, queue and Canary regressions:

```text
87 passed, 2 documented historical tests deselected, 3 subtests passed
```

The two deselections are:

1. a legacy metric-free third Canary validation that is not the V13 route;
2. an M3 assertion that current source bytes must equal the permanently
   superseded V12 frozen source manifest.

The V12 contract and its historical evidence were not modified.

Meta control-plane regressions:

```text
29 passed
```

`compileall` and `git diff --check` passed.

## Verdict

`PASS - P0=0 / P1=0 / P2=0`

G4 does not authorize a Pilot. G5 must freeze and qualify the Main-grade
bounded compositional executable profile before any V13 run.
