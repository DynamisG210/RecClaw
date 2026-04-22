"""Popularity-aware negative sampler stub for local RecClaw experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch


class PopularityAwareNegativeSampler:
    """Sample items from a popularity-shaped negative distribution."""

    def __init__(
        self,
        popularity: Union[torch.Tensor, Sequence[float]],
        alpha: float = 0.75,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        weights = torch.as_tensor(popularity, dtype=torch.float).flatten().clamp_min(0)
        if weights.numel() == 0:
            raise ValueError("popularity must not be empty.")

        shaped = weights.pow(float(alpha))
        if shaped.sum() <= 0:
            shaped = torch.ones_like(shaped)
        self.probability = shaped / shaped.sum()

    @property
    def num_items(self) -> int:
        return int(self.probability.numel())

    def sample(
        self,
        sample_shape: Union[int, Sequence[int]],
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        shape = (sample_shape,) if isinstance(sample_shape, int) else tuple(sample_shape)
        target_device = (
            torch.device(device) if device is not None else self.probability.device
        )
        total = int(torch.tensor(shape).prod().item())
        samples = torch.multinomial(
            self.probability.to(target_device),
            total,
            replacement=True,
        )
        return samples.reshape(shape)
