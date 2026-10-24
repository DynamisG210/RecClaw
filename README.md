# RecRSI

**Mechanism-Guided Discovery of Recommendation Algorithms**

RecRSI organizes recommendation research around mechanism hypotheses. It constructs an explicit mechanism space, implements and evaluates candidate algorithms, and uses mechanism-linked experimental feedback to guide the next research decision.

1. **MechSpace** identifies design choices and composition relations from the starting model and research literature.
2. **Research Line** develops hypotheses, constructs candidate programs, and selects experiments.
3. **Mechanism Feedback** connects measured outcomes to the design choices and evidence used in subsequent proposals.

The default task is SASRec on Amazon CDs and Vinyl. Pairwise graph recommenders use the `recbole-graph-fixed100-v1` adapter.

## Install and try

Download or clone this repository, open its root directory, and use Python 3.10–3.12. This CPU example runs without an API key or GPU:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[train,test]'
python -m pip check
recrsi demo --output ../recrsi-demo
```

The example creates four synthetic user histories and a 120-item catalog, builds fixed candidate sets, trains SASRec for two epochs, evaluates NDCG@5 and Hit@5, and reloads the checkpoint to verify its scores. Read `../recrsi-demo/DEMO_RESULT.json` and `../recrsi-demo/baseline/worker/LEARNING_CURVE.jsonl`. The research loop below adds model-generated hypotheses, candidates, and feedback.

## Run a research task

Use a Linux x86-64 CUDA host for the research loop. Complete [host setup](docs/host-setup.md) before running generated candidates. For CUDA, install the training extra in a CUDA-capable environment rather than preinstalling the CPU-only PyTorch wheel above.

```bash
recrsi init --output ../my-research
```

Edit `environment.json` to select the training interpreter, RecBole directory, NVIDIA libraries, shared resource-lock directory, four CPU cores, GPU UUID, and private API configuration. Edit `inputs/binding.json` to set the dataset name, model configuration, and training recipe. Put the four TRAIN/DEV/catalog files described in [task format](docs/task-format.md) in `../my-data/`. For the default task these are `cds.train.inter`, `cds.dev.inter`, `cds.user`, and `cds.item`.

For a new sequential task, start with a TSV containing `user_id`, `item_id`, and integer `timestamp` columns, then construct those files:

```bash
recrsi split --interactions ../interactions.tsv --output ../my-data \
  --test-output ../heldout/cds.test.inter --dataset cds --max-length 50
```

The splitter sorts each user's events by time, retains the first occurrence of each item, and uses the last two distinct items for DEV and TEST. Users with fewer than four distinct items are counted and omitted. `SPLIT.json` records the input hash and preprocessing counts. Existing frozen splits can be supplied directly. Match `MAX_ITEM_LIST_LENGTH` in the binding to `--max-length`.

Create `../literature.json` with the research literature for your task:

```json
{"documents": [{"title": "Self-Attentive Sequential Recommendation", "year": 2018, "summary": "Self-attention models sequential item dependencies. Add the evidence relevant to your task here."}]}
```

Run the following in the training environment so asset construction uses the same RecBole version as evaluation:

```bash
recrsi doctor --environment ../my-research/environment.json
recrsi assets --project ../my-research --data-dir ../my-data \
  --literature ../literature.json --description "Sequential recommendation on CDs"
recrsi baseline --project ../my-research --output ../my-baseline --device cuda
recrsi check --config ../my-research/task.json
recrsi prepare --config ../my-research/task.json --output ../my-run
recrsi inspect ../my-run
recrsi run ../my-run
```

`assets` obtains RecBole's actual token mapping and samples 99 fixed negatives per DEV target, excluding TRAIN interactions. `baseline` trains the starting model, verifies its checkpoint, and generates the results, physical cost ledger, and task facts used by search. A CPU baseline is available through `--device cpu`; its receipt records zero GPU time and measured elapsed time. Use the same selected training device and recipe across the comparisons you run.

`check` validates data, source, baseline, protocol, and accounting identities. `prepare` captures the runtime and task configuration in a fresh directory. `inspect` checks controller assembly. `run` executes mechanism-space construction, research proposals, candidate qualification, training, evaluation, and feedback. Resume in the original directory with `recrsi run ../my-run --resume`.

The method profile uses search seeds `41000 + repeat`, training seeds `51000 + repeat` (repeat 1–3), up to 36 research rounds, 12 candidate evaluations, 36 training admissions, and 7 reference USD shared by space construction and search. Model routing and accounting rates are recorded in the prepared configuration. The provider must expose `gpt-5.6-luna` and `gpt-5.6-terra`; see [host setup](docs/host-setup.md).

## Read the outputs

Run outputs are under `my-run/task/research_line/runs/<run_id>/`:

- `RESULT.json`: terminal search result and retained frontier.
- Candidate directories: exact source, qualification result, learning curves, checkpoint, and validation metrics.
- MechSpace, hypothesis, selection, and feedback records: research decisions linked to observed outcomes.
- API and GPU ledgers: requests, reference API cost, measured worker time, failures, and unresolved work.
- `PAUSED.json`, when present: the saved condition requiring reconciliation before continuation.

NDCG@5 and Hit@5 use fixed 100-item DEV candidate sets. Search uses TRAIN and DEV; keep TEST separate for final evaluation. Preserve all search repeats, including negative outcomes, when comparing discovery quality and cost.

## Development and distribution

```bash
python -m pytest tests -q
```

Tests cover configuration, native CPU training/evaluation/checkpoint reload, response parsing, watchdog termination, and controller assembly. CI runs the suite and standalone example on Python 3.10 and 3.12.

See [task files](docs/task-format.md), [host setup](docs/host-setup.md), and [anonymous source export](docs/anonymous-release.md). Code is distributed under the [MIT license](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) accompany the included SASRec implementation. The package and CLI are named `recrsi`.
