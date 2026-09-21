# Source identity, deployment and verification scope

Snapshot date: **2026-09-21**. Purpose: current-code reading and architecture communication for a course presentation. Not a final release, a model checkpoint release, a training launch or a new scientific comparison.

## What was captured

The main source is the owner's accepted `runtime_metric5_accepted/package`, independently compared file by file against deployed `graph/formal_runtime_metric5_v1` via read-only SSH. The accepted archive SHA256 is `a08827b51575a0da3d1f8d8178198303461257ed5d928a55785cde0e035abad6`. Exact file identities, the source closure hash, excluded categories and publication changes are in [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json).

This is a selected source package, not a copy of a dirty worktree or the repository's old default HEAD. No experiment directory, data, checkpoint, generated private search space, paid Provider request, private literature corpus or credential file is included.

Three retained core files contain historical private deployment locators. Those strings alone were replaced with inert public placeholders in `standalone.py`, `resource_scheduling.py` and `canary_broker.py`; the manifest records original and published hashes. These historical compatibility/profile locators are not portable deployment settings. All other captured source files retain their bytes. Private runtime release observations/locks and historically calibrated policy checkpoints were excluded, not replaced with fabricated defaults. The snapshot is not an exact runnable clone of private deployment state.

## Versions must remain distinct

| Source line | What this snapshot establishes |
|---|---|
| New graph formal `metric5` | Source deployed and publication files hash-matched; primary contract is NDCG@5. Owner's latest inspected record still awaited actual formal roots; deployment is not completed formal search or effectiveness. |
| Existing graph development `search_source_v2` | Older frozen NDCG@10 line; existing runs are not upgraded by publishing this directory. Its results cannot be relabeled NDCG@5. |
| MoRec active source | Five AutoSpace/feedback modules were read from the actual source and match RO01's original execution-binding hashes. Compared core has 308/309 equal original source hashes; distinct `campaign.py` is included under `variants/morec/core/`. |
| SASRec / BERT4Rec / MultiVAE | Separate task/version histories. This publication does not assert identical deployment, rerun them or recalculate their results. |

The core comparison is before publication filtering and the three public locator substitutions and covers 309 common source/resource files, not a claim of whole deployment equivalence. MoRec variant modules are for reading; do not inject them into the graph runtime. Other adapter dependencies, private calibrated policy artifacts and task contracts remain external.

The documentation intentionally contains no selected-seed leaderboard or new superiority claim. Formal comparison, final-fit and controlled-test reporting remain with the experiment owners. RO/RH labels refer to experiment methods, not guaranteed rankings.

## Actual verification

See [VALIDATION.json](../VALIDATION.json) for the final local check record.

- All published graph source origins compared with remote SHA256; publication hashes independently checked locally.
- Python AST parsing, JSON parsing, manifest coverage, maintained document links and fences, private locator/key-pattern checks.
- Linux/WSL import of the real Research Line and graph adapter; schema validation, source eligibility positive/negative fixture, and real feedback projection using synthetic curve records, including missing-curve handling.
- No Provider request, GPU job, new training, experiment-state mutation or heldout access.

The first native Windows core import failed because `fcntl` is Linux-only. A first Linux import with unpinned jsonschema failed the existing BL-ICF exact-version contract; using `jsonschema==4.26.0` and `rfc8785==0.1.4` resolved it. These checks establish inspectability and the demonstrated interfaces, not complete GPU runtime availability or scientific gains.

Full task execution additionally requires external modules such as the common API/evaluation/resource ledgers, worker confinement and task-specific bindings. Their interfaces remain visible in the captured code; private deployment closures/configurations are not published. No claim is made that `run_graph_ro_main.py --execute` works from this snapshot alone.
