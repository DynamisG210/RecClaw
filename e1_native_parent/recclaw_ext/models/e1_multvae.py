from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.utils import InputType


class FreshCandidateModel(GeneralRecommender, AutoEncoderMixin):
    """Compact user-wise multinomial VAE with the RecBole model ABI."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        hidden_sizes = tuple(int(value) for value in config["mlp_hidden_size"])
        if hidden_sizes != (384,):
            raise ValueError("E1 parent requires mlp_hidden_size=[384]")
        statistics_dimension = int(config["latent_dimension"])
        if statistics_dimension != 256:
            raise ValueError("E1 parent requires latent_dimension=256 for z=128")

        self.hidden_dimension = hidden_sizes[0]
        self.latent_dimension = statistics_dimension // 2
        self.input_dropout = float(config["dropout_prob"]) * 0.25
        self.anneal_cap = float(config["anneal_cap"])
        self.total_anneal_steps = int(config["total_anneal_steps"])
        self.update = 0

        self.build_histroy_items(dataset)
        self.encoder_hidden = nn.Linear(self.n_items, self.hidden_dimension)
        self.encoder_stats = nn.Linear(
            self.hidden_dimension,
            2 * self.latent_dimension,
        )
        self.decoder_hidden = nn.Linear(
            self.latent_dimension,
            self.hidden_dimension,
        )
        self.decoder_output = nn.Linear(self.hidden_dimension, self.n_items)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.encoder_hidden,
            self.encoder_stats,
            self.decoder_hidden,
            self.decoder_output,
        ):
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)

    def forward(self, rating_matrix):
        corrupted = functional.dropout(
            rating_matrix,
            p=self.input_dropout,
            training=self.training,
        )
        normalized = functional.normalize(corrupted, p=2, dim=1)
        hidden = torch.tanh(self.encoder_hidden(normalized))
        mu, logvar = self.encoder_stats(hidden).chunk(2, dim=1)
        if self.training:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            latent = mu
        decoded = torch.tanh(self.decoder_hidden(latent))
        logits = self.decoder_output(decoded)
        return logits, mu, logvar

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        clean_rating_matrix = self.get_rating_matrix(users)
        logits, mu, logvar = self.forward(clean_rating_matrix)
        reconstruction = -(
            functional.log_softmax(logits, dim=1) * clean_rating_matrix
        ).sum(dim=1).mean()
        kl_divergence = -0.5 * (
            1.0 + logvar - mu.pow(2) - logvar.exp()
        ).sum(dim=1).mean()
        if self.total_anneal_steps > 0:
            kl_weight = min(
                self.anneal_cap,
                self.update / float(self.total_anneal_steps),
            )
        else:
            kl_weight = self.anneal_cap
        self.update += 1
        return reconstruction + kl_weight * kl_divergence

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        logits, _, _ = self.forward(self.get_rating_matrix(users))
        rows = torch.arange(items.shape[0], device=self.device)
        return logits[rows, items]

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        logits, _, _ = self.forward(self.get_rating_matrix(users))
        return logits.reshape(-1)


__all__ = ["FreshCandidateModel"]
