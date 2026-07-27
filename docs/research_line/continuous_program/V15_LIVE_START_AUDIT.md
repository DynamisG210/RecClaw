# V15 Live Start Audit

## Verdict

`PASS` — P0=0, P1=0, P2=2. V15 may start once on the frozen root and seed.

This audit authorizes only the five-round-per-Arm V15 chain Pilot. It does not authorize Main or reinterpret the run as an Effect Pilot.

## Frozen identity

- Selected source commit: `10e8fca7768a7aeb295973925df050bf07ec4fd3`
- Contract SHA-256: `afbceb0fec4e3871793fa061a2159a10bd9c709ca025d232e13d2ccb008de801`
- Contract content digest: `b7ce48653f91c42f5c83ed0fa925bbc85c5aeff6e9d21ddf4d3236dd7a7611d0`
- Search seed: `9217`
- Rounds per Arm: `5`
- Executable profile: `f748b4b4b3103eadf2c3c262c14c4b779893255f2318bd99ebbf9f9eeee47536` with 66 executable semantics
- Meta policy bundle: `95923b800e89c4c4bfb994b5aa8a16069ac8429af056b456f01b42af4ab33744`
- Training runtime: `TRAINING_RUNTIME_RELEASE_V8`, digest `cbff7492788bb1616e50fb3b947454b8af6f078abbb3df32d941b8489250f80b`
- Backend: one `NVIDIA GeForce RTX 4090` hardware class, driver `550.76`
- Research LLM: laboratory API `gpt-5.4`

## Independent checks

- Frozen contract build and independent verification: PASS.
- Scientific Attribution Gate: PASS, P0=0/P1=0.
- Backend conformance audit: PASS, P0=0/P1=0.
- Local V15/V14/training-runtime regression: 20 passed.
- Remote backend, filesystem, cross-Arm and Original regression: 6 passed.
- Final-digest fixed BPR, LightGCN and compositional canaries: PASS 3/3.
- Remote compile/import and live runtime validation: PASS.
- Pinned Original materialized the six exact Main blobs and matched every frozen SHA-256.
- Arm A is `PinnedOriginalMainAdapterV1` at Main commit `2d8c881354e1b536a6c66d7dfbb977e0c5090e50`.
- B and C have identical Research controller, Producer, Router, Meta, BL-ICF and common-guard policy projections; their declared treatment difference is `NullEvidencePortV1` versus `EvidenceGuardPortV1`.
- No V15 process was running and the frozen output root was absent.
- The selected physical GPU was sampled three times over 40 seconds at 0% compute utilization. Its existing external allocation leaves about 2.0 GiB free; all fixed mechanisms used about 0.4 GiB additional memory.

## P2 limitations

1. The host lacks an unprivileged private mount namespace. The frozen backend uses package-owned execution, run-private cwd/environment, and before/after shared/sibling-root content hashes.
2. Another process retains memory on the selected GPU. The Pilot must remain supervised; external contention may cause a typed frozen-budget rejection but may not justify widening the budget or retrying V15.

## Start decision

GO. Use the exact frozen contract, seed, roots, V19 checkpoint, V8 runtime and laboratory `gpt-5.4` broker. No threshold, budget, retry, fallback or treatment change is permitted after start.
