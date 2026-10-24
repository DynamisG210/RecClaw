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

The diagnostic checks finite loss, projection gradients, agreement between
predict/full-sort interfaces, and equality of initial parameter tensors. This
guards an observed confound: adding the projection before recursive weight
initialization changes the RNG stream even under the same seed. The historical
root/objective pair and combination/score-only pair can have matched parameters
within each pair without having shared parameters matched across the pairs.

Historical DEV gains are descriptive. The missing score-only result prevents
computing the interaction. Completing that one cell does not retrospectively
create a prospective matched-initialization experiment or independent search
replication. Preserve the legacy root's ABI-equivalence provenance; do not
relabel it as a newly trained control. Future independent matched blocks require
an explicit prospective initialization/data/worker/budget contract, retaining
all outcomes and selecting seeds before observing results.
