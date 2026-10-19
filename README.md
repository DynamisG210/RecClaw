# RecRSI

**Mechanism-Guided Discovery of Recommendation Algorithms**

RecRSI takes a recommendation model, a frozen task environment and a finite experiment budget. It constructs mechanism hypotheses, implements candidate algorithms, trains and evaluates them, and uses the observations to guide subsequent research.

The final research path is:

1. **MechSpace**: source-grounded Genesis and Critic calls followed by deterministic assembly of mechanism objects, implementation requirements and composition relations.
2. **Research Line**: Mechanism Composer, Lineage Refiner, Falsification Designer and Frontier Architect construct hypotheses and complete candidate programs. The graph adapter supports explicit candidate-local mechanism definitions and measured construction parents. Two committed programs are compared by an independent selector; selection preserves their exact source bytes.
3. **Mechanism Feedback**: qualification, executed training signals, validation outcomes and source-linked contrasts update the next research context. Construction-parent, original-root and current-frontier comparisons remain distinct. Negative results and unresolved explanations are retained. The search policy is explicit and frozen; this release does not claim online policy learning.

## Distribution

This repository contains executable source, runtime prompts/schemas, installation metadata, focused tests and a task configuration example. Experiment logs, paid responses, credentials, datasets, checkpoints, paper drafts and coordination documents are external inputs or generated outputs.

The packaged end-to-end adapter is the finalized **NGCF / Electronics, fixed-100-candidate** research path. Its original protocol identifiers and seeds are retained. The shared mechanism and research components are included; other paper environments require their own task adapters and are not exposed as interchangeable command-line presets.

## Install

Production execution requires Linux x86-64, Python 3.10–3.12, a CUDA-capable PyTorch environment, an NVIDIA GPU with capacity for the 9 GiB worker allocation, and four assigned CPU cores. The native worker boundary uses Linux namespaces, Landlock and an unprivileged worker identity. The operator must provision the required `sudo -n` launcher access, readable runtime/data paths and NVIDIA driver-library directory. Generated programs run inside that boundary.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[train,test]'
recrsi --help
```

The training extra pins PyTorch, NumPy, SciPy and the RecBole source revision used by the worker. Set `recbole_root` to the installed `site-packages` directory containing `recbole/`, or to a checkout of the pinned RecBole revision. A separate controller environment is supported through `worker_python`.

## Prepare and run

Copy `examples/task.json` to a location outside Git and replace its paths. Relative configuration paths resolve against that configuration file. Supply the existing frozen task bundle:

- `binding`: task identity, training recipe, evaluation contract, hashed train/dev/user/item files, catalog mapping and fixed candidate pool.
- `facts` and `literature`: the common task facts and literature cards used by the research context.
- `root_source`, `root_binding`, `root_calibration`, `root_gpu_ledger`, `root_mapping`: the original model and its completed, paired-seed baseline measurement, actual cost ledger and ID mapping. `prepare` checks source, data, seed and baseline identities; it does not manufacture or rerun a baseline.
- `worker_python`, `recbole_root`, `driver_root`, `resource_slots`, `cpus`, `gpu_uuid`: the actual runtime and shared host allocation. The initial GPU UUID identifies the task environment; each physical stage selects an available host GPU through the shared capacity allocator and records its actual UUID. All runs on a host must use the same canonical resource-lock directory.
- `api_config`: a private HTTPS endpoint configuration supporting the frozen model identities `gpt-5.6-terra` and `gpt-5.6-luna`, using the Responses and chat-completions interfaces. Its format is `base_url: https://YOUR_ENDPOINT/v1` and `api_key: YOUR_KEY`, on separate lines. Keep it outside the repository and worker-readable data roots. All nonembedding calls use this explicit endpoint2 route.

```bash
recrsi prepare --config /path/to/task.json --output /path/to/new-run
recrsi inspect /path/to/new-run
recrsi run /path/to/new-run
# After reconciling an interruption, retain the original directory:
recrsi run /path/to/new-run --resume
```

`prepare` copies the released runtime into a new directory and binds its source hashes, interpreter, installed dependency identities and task inputs. `inspect` assembles the actual entry under a read-only audit boundary: no provider request, worker launch or resource claim. `run` performs MechSpace generation, research selection, source qualification, training, evaluation and feedback under the common native ledgers. A completed run is read idempotently; unresolved dispatches remain subject to native reconciliation, even with `--resume`.

The retained protocol has three paired repeat identities, up to 36 research rounds, 12 effective candidate evaluations and 36 training admissions. MechSpace and search share a budget of 7 reference USD under the original fixed accounting rates. Reference prices are experimental accounting inputs, not current provider price quotations. Failures, negative/no-op results, paid bytes and unknown usage stay in the run's accounting. Search receives TRAIN/DEV evidence; held-out TEST is not a tuning input or a command in this search release.

Outputs are under `new-run/task/research_line/runs/<run_id>/`: MechSpace artifacts, committed alternatives and decisions, source-linked evidence, campaign state, candidate source, learning curves, checkpoints, validation metrics and API/GPU ledgers. The terminal result is `RESULT.json`, or `PAUSED.json` when reconciliation is required.

## Verification and interpretation

```bash
python -m pytest tests -q
```

The CPU integration fixture performs real optimization, fixed-100 scoring and exact checkpoint reload on synthetic data. Other checks cover immutable program selection, actual-parent contrasts and read-only inspection. Installation and the native controller/worker bootstrap are checked separately on Linux. These checks establish software behavior; they are not recommendation-quality measurements. A new paid-provider/CUDA research campaign is not part of the packaging test.

Scientific claims must use the accepted common-protocol evidence for each environment. Mechanism prose and static source differences are hypotheses or observations, not component-wise causal proof. Shared-host timings are not exclusive GPU timings, and the accelerated evaluator's validated metric agreement does not imply universal bitwise score equality. This repository makes no universal-best or cross-version pooled superiority claim.
