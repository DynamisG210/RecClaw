# V13 Live Start Audit

## Verdict before execution

`PASS — P0=0, P1=0, P2=3`

The live audit verified the frozen V13 identities and allowed the exact
five-round command to start. The later Provider rejection was not present in
the local JSON Schema validator and is recorded separately in the V13 result
audit.

## Resolved identities

- worktree: `/root/projects/RecClaw_static_pilot`
- branch: `feat/research-line-meta-v17-pilot`
- HEAD: `c4be6c4e8c5a1fe064f24f0571792644cde48f76`
- HEAD tree: `cd5c81972343f38c7cf0e0b1d9e5964f732c3725`
- upstream: none configured
- source-freeze commit: `a0956c64dc96fa61a2cf34039111ea0075d04f38`
- source-freeze tree: `fdf048b91ad4ddff22e1b838fe19313cb1177053`
- V13 contract SHA-256:
  `15ee85e50d1b7ca3143276e775faa369d01861a638b6a8507114288d93d3d2ee`
- V13 contract content digest:
  `f9c16d06604b1789222a48e5af11c004d6a2bad5aa8cd7512d3a609b5dbb029f`
- source manifest:
  `620106b753d12b7f9edd296d728b525fde2716f41383c1289d338bb3c101dc3b`
- G7 gate:
  `262c39b4af9f34aaaaa6fcb3ca7bd25927c0bf3e36f78295a63eeeea5272c970`

The diff from the source-freeze commit to HEAD contains G8 records and
reports only. It does not alter the authoritative source projection.

## Runtime and isolation checks

- all 99 activated V13/Lab API/training/runtime tests passed;
- focused G7/G8 checks and Python compile/import preflight passed;
- the output root was absent and no V13/Pilot/training process was running;
- runtime imports resolved to the frozen worktree source projection;
- Arm A resolved through `PinnedOriginalMainAdapterV1` to Main commit
  `2d8c881354e1b536a6c66d7dfbb977e0c5090e50`;
- the dirty working-copy `scripts/agent.py` was outside the source manifest
  and unreachable from the V13 factory;
- B/C shared the same Research source, Meta V18 checkpoint, initial state and
  non-Guard policy; their only treatment difference was the EvidencePort;
- Guard-private fields were rejected from Producer, Router and Meta inputs.

## Pre-existing worktree boundary

The worktree contained older M6/V10/V12 edits and artifacts. They were not
staged, imported, hashed into V13, copied into the authoritative projection or
modified by this audit.

## Declared P2 limitations

1. Fast residual was disabled because the ML-1M task context is outside its
   frozen support envelope.
2. Several axes use generic transfer without axis-specific learned
   coefficients.
3. Slow-policy concentration required observation in a development Pilot.
