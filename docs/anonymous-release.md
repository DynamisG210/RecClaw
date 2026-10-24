# Anonymous source distribution

Create an anonymous source archive from the reviewed commit:

```bash
python tools/export_source.py --output ../recrsi-source.zip
```

The exporter includes tracked source, tests, examples, documentation, and
licenses. It omits Git history and hosting automation, fixes ZIP timestamps,
and checks for credentials and local workstation paths. It exports committed
bytes and refuses a dirty worktree so the archive corresponds to a reviewable
revision. Inspect the archive before uploading it to an anonymous repository.

Create the anonymous repository with a new history and an anonymous account;
do not push the original Git history or copy issues, PR discussions, release
metadata, or remote configuration. Review the hosting account and archive
description as well as file contents. Preserve third-party licenses and author
attribution in the SASRec template. These identify the upstream implementation.

Keep datasets, credentials, model responses, checkpoints, run logs, and paper
manuscripts outside the source distribution. Reproduction data or experiment
results can be supplied separately after checking their provenance and metadata.
