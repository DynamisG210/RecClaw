"""Task-defined RH directions for the shared E2 external mutation boundary.

This is vocabulary/attainability adaptation, not a new routing algorithm. The
constructor and loader remain frozen; permitted update-kernel changes retain
their actual meaning instead of disguising batch changes as loss reweighting.
"""
from __future__ import annotations

HINTS = {
    "tune_dropout_wd": "Jointly adjust model dropout and weight decay applied in the permitted optimizer-update kernel; do not alter optimizer construction.",
    "tune_embedding_dim": "Adjust model-internal embedding width while preserving the external model/evaluator interface.",
    "tune_num_layers": "Adjust model-internal depth; preserve the required input/output interface.",
    "tune_num_heads": "Adjust model-internal attention heads with compatible dimensions.",
    "tune_seq_len": "Adjust how much of the provided history the model internally uses. Loader maxlen200, ordering and validation candidates stay fixed.",
    "change_loss_function": "Change the learning objective in models.py or the marked training-update kernel; validation and checkpoint selection stay fixed.",
    "change_architecture": "Change the recommendation architecture inside models.py while preserving shared data/evaluation and the external call interface.",
    "change_pooling": "Change sequence aggregation/pooling within the model, not validation scoring or candidate order.",
    "add_features": "Derive additional features from the already available train-only inputs inside the allowed model/update code; no new data or loader changes.",
    "tune_lr": "Change the effective step learning rate inside the permitted optimizer-update kernel. Keep optimizer construction/class and all outer training/evaluation code unchanged.",
    "tune_dropout": "Adjust dropout within the model implementation, leaving the external training recipe unchanged.",
    "tune_weight_decay": "Change weight decay within the permitted optimizer-update statements; do not replace the optimizer or edit its frozen constructor.",
    "add_lr_scheduler": "Implement a step learning-rate schedule solely in the permitted update kernel. No changes to epochs, evaluation cadence, checkpoint selection or optimizer construction.",
}
COMPOSITES = {"tune_dropout_wd": ["tune_dropout", "tune_weight_decay"]}
EXCLUDED = ("tune_lr_batch_scheduler", "tune_batch_size")

SCOPE_NOTE = """
E2 task-specific RH arms retain native Thompson/group/jump/retuning decisions.
Loader batch-size changes and the coupled LR/batch/scheduler arm are unavailable:
the shared external contract freezes loader construction/behavior. Do not
silently relabel gradient accumulation or loss weights as a batch-size change.
LR, weight-decay and scheduling arms mean changes to the explicitly mutable
optimizer-update kernel only; frozen optimizer construction is not editable.
Model-internal context selection is allowed; loader maxlen200 stays fixed.
No import side effects, alias tricks or modifications to frozen objects outside
the permitted update execution may bypass the shared mutation contract.
""".strip()
