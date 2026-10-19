"""Cold-start templates for RecHarness.

Each template directory contains:
  train.py   — standalone training script (RecHarness evolves this as best.py)
  predict.py — implements predict(user_id, history, candidates) -> list[float]
               called by evaluate_val_fast / evaluate during each trial
               (not required for regression tasks like gr)

Available templates
-------------------
popular   : Item popularity baseline.  Fast, trivially reproducible.
sasrec    : SASRec (Self-Attentive Sequential Recommendation). Strong baseline.
gru4rec   : GRU4Rec.  RNN-based, good for long sequences.
bert4rec  : BERT4Rec. Bidirectional Transformer with cloze-style pre-training.
perdataset: Per-dataset SASRec, one checkpoint per Amazon Reviews dataset.
gru4rec_perdataset   : Per-dataset GRU4Rec.
bert4rec_perdataset  : Per-dataset BERT4Rec.
nextitnet_perdataset : Per-dataset NextItNet.
hstu_perdataset      : ID-only HSTU-style per-dataset sequential recommender.
gr        : GR (Generative Regression) on KuaiRec watch-time prediction.
d2q       : Duration-Deconfounded Quantile baseline for KuaiRec.
tpm       : Tree Progressive Model baseline for KuaiRec.
"""
from __future__ import annotations

import os

TEMPLATES: dict[str, str] = {
    "popular":    "popular",
    "sasrec":     "sasrec",
    "sasrec2":    "sasrec2",
    "gru4rec":    "gru4rec",
    "bert4rec":   "bert4rec",
    "gr":         "gr",
    "d2q":        "d2q",
    "ks_d2q":     "d2q",
    "tpm":        "tpm",
    "perdataset": "perdataset",
    "sasrec_perdataset": "perdataset",
    "gru4rec_perdataset": "gru4rec_perdataset",
    "bert4rec_perdataset": "bert4rec_perdataset",
    "nextitnet_perdataset": "nextitnet_perdataset",
    "hstu_perdataset": "hstu_perdataset",
}

_HERE = os.path.dirname(__file__)


def get_template_dir(name: str) -> str:
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown cold-start template '{name}'. "
            f"Choose from: {sorted(TEMPLATES)}"
        )
    return os.path.join(_HERE, TEMPLATES[name])
