# Dependencies and attribution

RecClaw is the research orchestration and task-integration code in this snapshot. It does not claim authorship of the recommender algorithms, third-party training libraries, datasets, or external language models.

- **RecBole** supplies recommender abstractions, training/evaluation infrastructure and existing model implementations used by external task environments.
- **PyTorch**, **NumPy**, **SciPy** and, for applicable environments, **Transformers** supply model computation and numerical functionality.
- **jsonschema**, **referencing**, **rfc8785** and **PyYAML** supply schema validation, reference resolution, canonical JSON and configuration parsing. Some core paths also use **tiktoken** for token accounting.
- **SASRec, BERT4Rec, MoRec, MultiVAE, NGCF and LightGCN/LightGCN++** refer to external recommendation research/model families or the corresponding task integrations, not algorithms invented by this snapshot.

These dependencies are installed separately, not vendored here. Upstream project licenses and attribution requirements continue to apply. Original graph model files, third-party comparison systems, literature full texts and datasets are deliberately not redistributed in this snapshot. The offline example uses a minimal synthetic interface fixture, not a recommender implementation.

No repository-wide license was found in the checked base revision. This snapshot does not invent a new blanket license or relicense third-party work. Public source availability alone does not establish unrestricted reuse rights.

`requirements-inspection.txt` is sufficient for the Linux import/readout demonstration; it is not a training environment lock. Full execution depends on task-bound interpreters, GPU stack, RecBole compatibility, data contracts, resource/cost modules and OS confinement. Those private deployment bindings are excluded. External Provider credentials are supplied by a separately configured deployment, never by this repository.
