# V14 Recovery and Start Audit

## Verdict

`PASS — P0=0, P1=0, P2=3`

V14 is a fresh successor, not a V13 repair in place. It uses seed 9216, new
state and roots, a new source checkpoint, new contract, strict proposal
schema, Broker release, training release and Meta transport checkpoint.

## Closed root cause

The Provider-strict validator now requires every object property to be
required and rejects a drifted schema before any external call. The common
proposal schema requires:

- exact `mechanism_id` matching the typed composition;
- `original_priority` as a required nullable transport slot;
- a non-null Original priority for Arm A;
- null Original priority for Research Producers.

A treatment-free real `gpt-5.4` conformance probe passed both paths:
four Original proposals and one Research proposal, all executable
compositions, two physical calls and zero retries.

## Frozen identities

- source commit:
  `f5dbfbab671a9ec7d95e59598d5bb820b6980916`
- contract SHA-256:
  `190a47183722a9c2009cdaed62c43d662be3aa5ff4c086aa94fdfd47d72e4881`
- contract content digest:
  `44b7a81793de8a94514c28d7b910d6f27c938ddfc5fb218c6a2ffbcf595ab592`
- source manifest:
  `eac82b4ca28f7823bafb75f7258571db11263f2d4febc118552126ae69c92406`
- executable profile:
  `f748b4b4b3103eadf2c3c262c14c4b779893255f2318bd99ebbf9f9eeee47536`
- runtime profile:
  `8ec10230f7eda18424376855422200e5b5326483be0ec0042a99b386bf88d6e7`
- Meta V19 bundle:
  `95923b800e89c4c4bfb994b5aa8a16069ac8429af056b456f01b42af4ab33744`
- Broker release:
  `8b5adda11f94cdac4495b0587058df9fdd9afb79e05a048b8d857abd0548ee31`
- training release:
  `2910cbbb9e3c6f5b4a1725a05fe8c651d5c7f74546fe91bedc4b8dcee4ffdbc9`
- scientific gate:
  `6ab03c3a13766efddf9c96124d33880a1b6bfce938bed1a47a1c5e5a9faf19ce`

## Scientific equivalence

- A still uses exact pinned Main commit
  `2d8c881354e1b536a6c66d7dfbb977e0c5090e50`.
- A/B/C share the same 66-semantics BL-ICF profile and training release.
- B/C share Research code, initial state and unchanged V18 learned slow
  policy/router coefficients.
- V19 changes only the content identity required by the strict transport
  schema; no V13 outcome is used.
- B/C differ only at the EvidencePort and its legitimate downstream state.

## Verification

- V14 recovery and cross-component gate: 6 passed.
- strict Provider schema regression: 2 passed.
- activated engineering/scientific suite: 222 passed, 26 subtests passed.
- compile/import preflight: pass.
- training runtime validation: pass.
- frozen contract build and independent verification: pass.
- output root absent and no active V14 process before launch.

One legacy M5 fixture fails because that obsolete three-round harness does
not construct the validation task required by the current V13 admission
contract. V14 uses the typed task-queue orchestrator exercised by the passing
cross-component gate; the legacy harness is not in the V14 source or runtime
path.

The three inherited Meta P2 limitations remain declared. They do not lower a
P0/P1 gate.
