# R1/R2 Prefreeze Status

## Status

`BLOCKED_PREFREEZE`

The independently accepted Wave 2 source is the only source identity:

- commit: `56d3156f53d17d4d850368ce7a093fa368957d81`
- Git tree: `30b239b7ac85cd8de176d4b719b327ba02b96e0b`
- deterministic tree archive SHA-256: `f25d901a5493399c247f2e205a834c485ab52a32756cfc6fa97b97fe1d021ba7`
- independent Wave 2 gate receipt SHA-256: `fcb431c0ae8738f243ba10021770a2c507bee2b6d8454df622c2dc05700332eb`

No Provider, GPU, experiment, outcome, held-out, R1 root, R2 root, or database
was invoked, read, or created. No `R1_PREFREEZE_READY_RECEIPT.json` exists.

## Canonical artifacts

- `R1_R2_PREFREEZE_MANIFEST.json`: `cb1724a496a4ed6582e0babb7a5b3b3673844ad532be6d4db3a35b34b287a8bb`
- `PREFREEZE_BLOCKED_RECEIPT.json`: `d75547fcb073696b90ef928d121c2d95526efc23ff9e65eb6725b0d5bf7b15e1`

The manifest records every identity established without invention and leaves
unknown identities as JSON `null`. The fail-closed validator rejects it, as
required. The blocked receipt is the authoritative list of unresolved fields.

## Established provider evidence

The accepted tree contains
`LabApiCanaryBrokerV1.call_with_session` at file SHA-256
`bfcf24f562ee92f5b3570907b0a6322fb92e6de5872b6f59005484649967992e`.
A sealed pre-outcome V18 contract, SHA-256
`b2db675e5524897f4db4ca84705b67385c76cd141c451082ddeb9776ff243f23`,
records gpt-5.4, endpoint digest
`8f6acc6b27581f6ae162c891bf65b5671f8c2d327f20cad9ba9f848fc45c38fc`,
`SINGLE_JSON_SCHEMA_NO_TOOLS`, and retry count zero.

That evidence does not close fresh OpenSpec R1. The only matching response
contract in the accepted path is the fixed-66 campaign schema
`campaign_proposal_response_v1.schema.json`, SHA-256
`37ff8e0e69ea33da13a0b6e1d45ff1707ab2a18731a58f376a30a39a9190e802`.
It cannot represent the required outside-profile OpenSpec proposal. Reusing it
would silently collapse R1 back into the frozen 66-entry space.

Credential material was not read. No credential digest was available in
authorized evidence, and no endpoint support check was performed.

## Frozen independent invariants

- model must be exactly `gpt-5.4`;
- proposal budget is 8 per side;
- A/B must bind one identical call-contract digest covering model,
  granularity, token budget, call count, prompt, tool policy, response
  contract, and failure rule;
- retry count is zero and no retry is permitted;
- zero-token infrastructure failure is not candidate-quality evidence;
- the accepted shared Implementer and Qualifier are byte-bound and
  origin-blind;
- manual candidate patching is forbidden;
- qualification remains `DEVELOPMENT_ONLY`;
- held-out remains absent;
- R1/R2 lineage, seed, outcome namespace, memory namespace, root, and database
  identities must all differ;
- all R1 result slots remain empty until real R1 receipts exist.

## Remaining intake

An independent pre-outcome freeze must supply:

1. a provider release and redacted credential identity verified to support
   exact gpt-5.4 without endpoint, model, protocol, or schema substitution;
2. the fresh OpenSpec prompt, response contract, tool policy, granularity,
   token budget, call count, and one identical A/B call-contract digest;
3. the exact runtime and dependency lock;
4. fresh R1 and R2 lineage, seed, root, database, outcome, and memory
   identities, including distinct R1 A/B identities;
5. pre-outcome missingness, threshold, and analysis identities.

Only after those values produce canonical manifest bytes accepted by the
validator may an independent worker emit `R1_PREFREEZE_READY_RECEIPT.json`.
That receipt is the sole legal prerequisite for R1 launch; this G status is not
a scientific gate approval.
