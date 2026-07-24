# M6R Independent Audit

Verdict: `PASS_WITH_NONBLOCKING_P2`

```yaml
P0: 0
P1: 0
P2: 1
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

## Independently verified evidence

- Frozen subject: base commit
  `d34c1ae99891442fc704de20897637fa075b4ccb`, tree
  `4a887e065179b77567aa3155c2a2260c6176e0d8`.
- Authorization, stop report, V3 failure record and pre-amendment roadmap
  hashes independently match the M6R manifest.
- Pilot V1/V2/V3 roots have zero diff from the hard-stop commit and retain
  tree IDs `55429fe884aae3829e8d836281bfbd5809975f52`,
  `e5896839d5e12a0c32aa7861dc376cb76b8d26bd`, and
  `dad92c4c9c65d23957d8f8b880fe7f12cb2eb191`.
- The fake release remains unchanged:
  runtime `0a616ee205f494a161e7b10424181f37247347de8c5d714328dfa2f3f1ffb41d`,
  source `038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd`,
  common projection
  `97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347`.
- The current training release validates with no failure:
  `9401d9c5f5f096057d183cf0e4dc4f2f76bbc85030099d17646ecdad5e8ad375`,
  ABI `recclaw.package-owned-search-training-runner.v1`, Python package
  lock `rfc8785=0.1.4`.
- Independent clean-environment reruns passed: full experiment suite
  `112/112`; targeted M6R suite `11/11`.
- V7 is the only retained M6R result root. Its verifier passed with 12/12
  indexed artifacts exact-byte, artifact-tree projection
  `SORTED_RELATIVE_PATH_AND_SHA256_V1` digest
  `e5fe2cd129c374b1159107236fc83f26d32233c943a7dc839131ddfda80c2539`,
  SQLite integrity `ok`, exactly eight tables, one `FINISHED` claim and one
  ordinary execution debit.
- Exact V7 chain:
  binding `3a2d46379d6baef1ed870bd41a20b88c60f3b1231c329a275b0fc1318e963ff4`;
  permit `232cd45f5a03136c18b2d737ca6d5558d63dda35a013cb26b44a76532e5aefe9`;
  claim `training-execution-claim:b98e92113acea0adb3b090e1dadd450de87debf6fb701843c9eb3bf252b7f130`;
  runtime binding
  `d5cdfe3f3d2df39fecb765bc13c4c0509ac7fda50926f59d246e8a7d1dad50e7`;
  start confirmation
  `a6a12aa90bd9ecf41fe98cbd40f61c31873bb1f50a8a53d84a07628ed6480aa9`;
  receipt `45b87496c72fa535eb40c0ab6a6c2b5a22663306a5f9c7d49ed79bdaaecb10fc`;
  raw output `6e07178739d3a9e2561ccadd0b92d62c96269fec31f978b5b90eb4ada93687ac`;
  common close `6852a2e048f840b03402d16b75c584464d7e7267bc653e82cc1d61a8b7d9843b`;
  raw envelope `dc51f9648020d905c7cd872e500da14b5ff40d02e0a8582f483a6011023730c7`.
- Canary accounting: search seed 9301, broker/LLM calls 0, ordinary
  execution 1, GPU device time 72,676 ms, wall time 72,676 ms, GPU cost
  20,188 microunits, NDCG@10 0.1581.

## P2-001 — Historical Pilot integration target

`RealPilotOrchestratorV1` retains the historical V3 seed 9203. It was not
executed and no sealed Pilot evidence changed. It must not be invoked as the
fresh Pilot. A later phase must first freeze a new Pilot contract/version,
unused seed greater than 9203, fresh root/store and fresh broker state.

This P2 is nonblocking for M6R and is recorded as M6R-R7.

## Conclusion

M6R satisfies the independent `P0=0/P1=0` gate. The result is engineering
feasibility only and grants no evidence authority. No fresh Pilot was run.
