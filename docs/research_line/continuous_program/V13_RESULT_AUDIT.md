# V13 Result Audit

## Verdict

`NON-GO — P0=0, P1=1`

V13 is permanently sealed. It must not be patched, rerun or reused. The
failure is a recoverable Provider response-schema compatibility defect, not a
scientific outcome and not evidence about Research Line effectiveness.

## Exact failure

The first physical `gpt-5.4` Broker request was rejected with HTTP 400 before
proposal generation:

> Invalid schema for response_format `recclaw_campaign_proposals`: within the
> proposal item, `required` did not include every property; `mechanism_id` was
> missing.

The package schema was valid JSON Schema 2020-12, but the Provider's strict
structured-output contract additionally requires every key in an object
`properties` map to appear in that object's `required` array. The same defect
also applied to `original_priority`, although the Provider reported the first
missing key only.

## Closed runtime state

- round:
  `search-round:f8b90c0b1d6d44e0227d4fb0a6e128c952411d535d9ca85024ace6f0edc29c02`
- opaque instance: `inst-49966f8f8adfea38372c2372`
- round index: 1
- terminal class: `BROKER_PROCESS_FAILURE`
- Broker calls: 1 physical, 0 successful, 0 retries
- proposal-attempt debit: 4
- billed/input/output tokens: 0/0/0
- ordinary executions: 0
- GPU cost: 0
- Guard calls: 0
- Search Memory updates: 0
- Meta observations: 0
- frontier updates: 0
- unclosed rounds: 0

The exactly-once `BROKER_FAILURE_CLOSURE_V1` closed the opened SearchRound
without retry, refund, fallback, Search Memory update or training.

## Evidence identities

- frozen contract content digest:
  `f9c16d06604b1789222a48e5af11c004d6a2bad5aa8cd7512d3a609b5dbb029f`
- Broker outcome digest:
  `f5a75dfa4c9cead421e9e0b0f4e20b9fbc2347e5f90cac0b416d588d93c910fe`
- failure closure digest:
  `9ae28a6ad2efcfb7a21a7f8ab8b638a5df1de0a9f0148bdd16d6390e95c30979`
- sealed root:
  `/root/projects/RecClaw_campaign_pilot_9215_v13`
- seal marker:
  `SEALED_V13_FAILURE.json`

## Attribution impact

No candidate, Guard decision, training result, Search Memory event, Meta
update or frontier state was produced. Therefore:

- no A/B/C treatment state was contaminated;
- no B-A or C-B effect can be estimated from V13;
- the failure does not count as a negative Research result;
- V13 cannot satisfy chain qualification.

## Authorized recovery

Apply one minimal provider-strict schema fix, add a regression validator,
rebind every affected content identity, rerun the affected gates, and perform
one treatment-free real Broker conformance probe. A successor Pilot must use
a new version, seed, contract, state and output root.
