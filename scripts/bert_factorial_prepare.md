# BERT4Rec scoring and objective decomposition

`bert_factorial_prepare.py` reconstructs the human-designated paper RO3 case
(historical RO08, round 7), using the original evaluation ledger and executed
source bytes. It verifies the combination and objective-only source hashes,
preserves every ledger row in its output, and generates the missing score-only
candidate by replacing smoothed CE with hard CE. It never launches training or
loads TEST data.

Required inputs are the archived BERT_RO08 directory, the original round-3
objective-only source, and a new output directory. Run with `--raw`,
`--objective-source`, and `--out`. Paths are supplied explicitly, rather than
embedding workstation or private experiment locations in the script.

`bert_factorial_cpu_check.py` accepts a JSON object on stdin containing
`recbole_root` and `sources` (root, objective_only, combination, score_only).
Run in the original Torch/RecBole environment. It uses synthetic 64-item data,
one CPU thread, and no optimizer step; this is not a FIT or quality measurement.
The model-config adapter is excluded explicitly because the diagnostic passes
a native mapping. Qualification of the actual wrapped worker remains necessary.

The diagnostic reports finite loss, projection gradients, predict/full-sort
agreement, and initial tensor equality for each source pair.

`bert_factorial_run.py --config CONFIG.json` executes one allocated cell using
the recorded native worker and shared CPU/GPU admission modules. The config
supplies paths and input hashes; the allocation supplies time and resource caps.
Train/dev bytes are checked against the recorded manifest before admission.
An existing ATTEMPT.json prevents automatic relaunch. Inspect the original
worker and persisted budget before diagnosing an interrupted attempt.

The output directory contains BINDING.json, ATTEMPT.json, native worker outputs,
GPU_BUDGET.json, ACTIVE_WALL.json, and RESULT.json or FAILURE.json. COST.json
retains the final metering snapshot. Queue time is outside the worker clock;
worker execution and unresolved reservations remain in the finite GPU ledger.
