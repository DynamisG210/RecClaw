# Q5 Q4 Outcome-aware KeyError root-cause diagnosis

Status: `DEVELOPMENT_ONLY_DIAGNOSTIC`

This diagnostic replays the sealed Q4 Outcome-aware materialization receipt
against the relocated selected candidate. It does not modify the sealed Q4
candidate, receipt, outcome, or source tree.

## Reproduction identity

- Evidence root: `results/research_line/q4_prospective_policy_comparison_20260803_01/arms/03_outcome_aware/`
- Selected candidate/spec digest: `146af84a1c273a5df82f1f5d3141f14874272367121c14344b5a9ec1d560609a`
- Candidate package digest: `546c792b7487a343fb7ccf93ba4c1dc883e1c3d2119bdad1bdaedd00dd38d498`
- Candidate root: `results/research_line/q4_prospective_policy_comparison_20260803_01/arms/03_outcome_aware/materialized/candidates/selected/innovation-candidate-b15a71ab8867711fa9299afa`
- Source tree digest: `0249f4900aa176bafb48c0c9b42b42afff2e2ca0b8946778e30af978fdf695f9`
- Protocol digest: `7f623fd953001f999e8b5d2657749f6a3ca86c7be5410a48bb8281241a258bbe`
- Qualification seed: `54301`
- Runtime: `RecBole` commit `7b02be5ec80a88310f2d04a27a82adfcbb5dc211`, existing `innovation_recbole_adapter._construct_runtime` and `MechanicalRecBoleAdapterV1.qualify`

## Observed root cause

The failure is candidate-local. During construction, the relocated
`FreshCandidateModel` calls `_build_train_histories`. Its `_get_time_field`
returns the dataset's declared `timestamp` field, but the mini qualification
fixture's `inter_feat` has no `timestamp` column. The candidate then performs
`inter_feat[time_field]` without checking that the declared field is present.
The shared adapter only constructs the model and correctly preserves the
result as a typed construction/interface failure.

The exact raw traceback from `_construct_runtime` is:

```text
Traceback (most recent call last):
  File "<stdin>", line 70, in <module>
  File "/root/projects/RecClaw_q5_foundation_execution/src/recclaw_core/experiments/helix_abc_v1/innovation_recbole_adapter.py", line 423, in _construct_runtime
    model = candidate_class(config, train_data._dataset).to(config["device"])
  File "/root/projects/RecClaw_q5_foundation_execution/results/research_line/q4_prospective_policy_comparison_20260803_01/arms/03_outcome_aware/materialized/candidates/selected/innovation-candidate-b15a71ab8867711fa9299afa/recclaw_ext/candidate.py", line 37, in __init__
    self._build_train_histories(dataset)
  File "/root/projects/RecClaw_q5_foundation_execution/results/research_line/q4_prospective_policy_comparison_20260803_01/arms/03_outcome_aware/materialized/candidates/selected/innovation-candidate-b15a71ab8867711fa9299afa/recclaw_ext/candidate.py", line 100, in _build_train_histories
    time_np = self._to_numpy(inter_feat[time_field])
  File "/root/projects/RecBole/recbole/data/interaction.py", line 135, in __getitem__
    return self.interaction[index]
KeyError: 'timestamp'
```

The existing qualifier result is `CONSTRUCTION / INTERFACE / KEYERROR`, with
no API, unit, smoke, mechanism, or effect authority. This is not a shared
consumer contract defect. The old candidate remains sealed and unchanged; no
candidate-specific repair, retry, replacement, or outcome was created.

## Regression boundary

The adjacent regression fixture intentionally models the same generic
candidate behavior and asserts that the existing qualifier preserves the
candidate-local failure class and stage. It does not alter the adapter or
weaken the shared contract.
