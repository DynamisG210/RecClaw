"""Fit-once spectral operators and the existing BPR coefficient training path."""
from __future__ import annotations

import torch
from recbole.trainer import Trainer

from recclaw_core.research_line.p4_runtime import p4_fit_mode


class P4OperatorTrainer(Trainer):
    def _build_optimizer(self, **kwargs):
        self.p4_fit_mode = p4_fit_mode(self.config.final_config_dict)
        if self.p4_fit_mode == "TRAIN_ONLY_PRECOMPUTE":
            if any(p.requires_grad for p in self.model.parameters()):
                raise ValueError("train-only precompute must store fitted coefficients as buffers")
            return None
        return super()._build_optimizer(**kwargs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True,
            show_progress=False, callback_fn=None):
        if self.p4_fit_mode != "TRAIN_ONLY_PRECOMPUTE":
            return super().fit(train_data, valid_data, verbose, saved,
                               show_progress, callback_fn)
        if not self.model._recclaw_p4_fit_completed:
            raise RuntimeError("P4 train-only initialization did not complete")
        self.eval_collector.data_collect(train_data)
        self.best_valid_score, self.best_valid_result = self._valid_epoch(
            valid_data, show_progress=show_progress)
        if saved:
            self._save_checkpoint(-1, verbose=verbose)
        return self.best_valid_score, self.best_valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        if self.p4_fit_mode != "TRAIN_ONLY_PRECOMPUTE":
            return super()._save_checkpoint(epoch, verbose=verbose, **kwargs)
        torch.save({
            "config": self.config, "epoch": -1, "cur_step": 0,
            "best_valid_score": self.best_valid_score,
            "best_valid_result": self.best_valid_result,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": None, "p4_fit_mode": self.p4_fit_mode,
            "operator_fit_completed": True, "optimization_epochs_completed": 0,
        }, kwargs.get("saved_model_file", self.saved_model_file), pickle_protocol=4)

    def resume_checkpoint(self, resume_file):
        if self.p4_fit_mode != "TRAIN_ONLY_PRECOMPUTE":
            return super().resume_checkpoint(resume_file)
        checkpoint = torch.load(resume_file, map_location=self.device, weights_only=False)
        if checkpoint.get("p4_fit_mode") != self.p4_fit_mode:
            raise ValueError("checkpoint P4 fitting mode differs from its declaration")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))
        self.model._recclaw_p4_fit_completed = checkpoint["operator_fit_completed"]
        self.best_valid_score = checkpoint["best_valid_score"]
        self.best_valid_result = checkpoint["best_valid_result"]
        self.saved_model_file = str(resume_file)
