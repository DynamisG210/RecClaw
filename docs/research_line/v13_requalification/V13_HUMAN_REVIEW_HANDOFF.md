# V13 Pilot-Start Handoff

## Current state

G0–G8 are locally PASS. G9 has not started.

- branch: `feat/research-line-meta-v17-pilot`
- source-freeze commit:
  `a0956c64dc96fa61a2cf34039111ea0075d04f38`
- source-freeze tree:
  `fdf048b91ad4ddff22e1b838fe19313cb1177053`
- frozen Pilot contract SHA-256:
  `15ee85e50d1b7ca3143276e775faa369d01861a638b6a8507114288d93d3d2ee`
- contract content digest:
  `f9c16d06604b1789222a48e5af11c004d6a2bad5aa8cd7512d3a609b5dbb029f`
- source-manifest digest:
  `620106b753d12b7f9edd296d728b525fde2716f41383c1289d338bb3c101dc3b`
- search seed: `9215`
- rounds: 5 per Arm
- output root:
  `/root/projects/RecClaw_campaign_pilot_9215_v13` — absent
- model: `gpt-5.4` through the laboratory API
- authority: `NONE`
- evidence class: `DEVELOPMENT_ONLY`

## Frozen identities

- executable profile:
  `d483faa471c3f26d321daa89c6946ab67a11a95d459d2d5c095c9da419ee5da1`
- runtime profile:
  `745b67509bb920a2345e99b8ef4c7ba3daffb8a3d62188ecc1e1978a3a002b87`
- Meta V18 bundle:
  `2d42ac7b9ced6057d3e20e19f1f470338ed00b0fcbc70e790bdce7fa44706024`
- Lab API release:
  `3fd4943a6771af1044a25ec7fff0cdb02a609c7e253ff6a16e7fcf2a56bb19d9`
- training release:
  `b0b1d54699fbdf0cffad11be2c0ec7ecb2d8c2c2fd221a98596d95ca2e0ded74`
- G7 gate result:
  `262c39b4af9f34aaaaa6fcb3ca7bd25927c0bf3e36f78295a63eeeea5272c970`

## Next authorized operation

Do not run Pilot as part of this handoff. When the user authorizes G9, first
verify the contract and output-root freshness, then execute exactly:

```bash
PYTHONPATH=src:. \
/root/projects/RecClaw_m6_training_runtime_v2/bin/python \
scripts/run_v13_pilot.py \
--contract docs/research_line/v13_requalification/V13_FROZEN_PILOT_CONTRACT.json \
--llm-api-config /root/projects/RecClaw_v2_0_Final_Reference/llm_api.md
```

After natural completion, perform the independent Pilot audit before assigning
GO or non-GO. Do not start V14, Main, M7 or M8 automatically.
