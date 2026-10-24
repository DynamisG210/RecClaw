# Same-incumbent history experiment utilities

These tools prepare and run a bounded continuation from a trusted native MoRec/CDs
checkpoint, retaining or resetting research experience. They are research scripts,
not a public CLI. The native checkpoint/runtime and task adapter must be supplied
externally; private paths, credentials, generated inputs and paper text are not
included in this directory. Unpickle only the owner's trusted artifacts.

## Recompute a completed stage

On the original Linux runtime, export existing evidence without launching work:

```sh
python -B export_completed_pair.py "$PAIR_HOME" "$ORIGINAL_HOME" > completed.json
python -B analyze_completed_pair.py completed.json analysis.json
```

The exporter reads the completed three-opportunity stages, native prepared-round
checkpoints, API events and GPU ledgers. It maps recorded call usage to unique
API events, including billed schema failures. The special first-batch recovery
artifact preserves the earliest paid proposals after a construction binding
failure. Unmatched successful API calls remain charged and explicitly listed.

The analysis separates candidate quality from the retained frontier, checks the
original ledger totals, excludes reused root cost from new GPU time, and compares
the four roles' actual initial request contexts. A mismatch in construction-parent
options is reported rather than silently normalized. Historical inputs and later
observations are not independent unseen evidence. Do not interpret candidate
mean differences as independent repeats or as TEST performance.

## Preparation and execution dependencies

`export_snapshot.py`, `prepare_pair.py`, `native_pair_probe.py`,
`capture_native_requests.py`, and `seed_paired_states.py` prepare and inspect the
state/input boundary. `prepare_runner_configs.py` prepares independent finite
resource manifests. Inspect their argument parsers before use.

`runner.py` composes the original native runtime with the isolated states,
construction parent and private budget binding. `admission.py` uses the shared
physical GPU/CPU admission system. `run_pair.py` serializes the two arms and refuses
to replay an already-started arm. The runtime directory contains the adopted
budget/watcher changes; the original runtime supplies the remaining modules and
`ORIGINAL_PATHS.json`. The deadline helper requires the existing host's authorized
noninteractive privilege configuration; these tools do not configure privileges.

The observed stage is fixed at chronological rounds 13–15, three valid candidate
evaluations per arm, with separately registered costs. Preparation does not grant
permission for a new paid stage. `repair_zero_call_runtime.py` is only for the
documented zero-call setup failure, never for rewriting a paid or completed run.
`--engineering-resume` requires diagnosis and preserved original state/ledgers;
it is not a general retry switch. Successful FITs must not be replayed.

For a prospective version, freeze the same construction-parent options before
either producer is called and verify the actual request difference boundary.
Existing completed inputs and results must remain unchanged.

`prepare_initial_contract.py completed.json new_contract_dir` prepares
`INITIAL_INPUT_CONTRACT_V2.json` and checks the actual recorded serialized prompts
through the prospective binding function offline. In a separately allocated new
pair home, the runner reads that manifest, binds identical parent options before
transport and checks source hashes and non-history fields. History, latest
observations, context digest and the architect's within-round generated portfolio
are the explicit allowed differences; parent availability is not. The new
contract rejects a `FIRST_PAID_BATCH_RECOVERY.pkl` from the old stage. Do not place
the manifest in a completed run. Offline transformed requests are not new paid
evidence or proof of a completed native run of this prospective version.

## Local checks

```sh
python -m unittest discover -s tools/research/history_value -p 'test_*.py'
python -m compileall -q tools/research/history_value
```

These checks verify preparation code and syntax only. The completed native export
is required for result/cost recomputation and actual request-boundary checks.
