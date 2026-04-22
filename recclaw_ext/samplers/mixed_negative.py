"""Mixed negative sampler stub for local RecClaw experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class MixedNegativeSamplerConfig:
    """Configuration for uniform/popularity-biased negative mixing."""

    num_items: int
    hard_negative_ratio: float = 0.5


class MixedNegativeSampler:
    """Sample negatives from a uniform and popularity-biased mixture."""

    def __init__(
        self,
        num_items: int,
        popularity: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        hard_negative_ratio: float = 0.5,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        if not 0.0 <= hard_negative_ratio <= 1.0:
            raise ValueError("hard_negative_ratio must be between 0 and 1.")

        self.num_items = int(num_items)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.popularity = self._build_popularity(popularity)

    def sample(
        self,
        sample_shape: Union[int, Sequence[int]],
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        shape = self._normalize_shape(sample_shape)
        total = prod(shape)
        target_device = (
            torch.device(device) if device is not None else self.popularity.device
        )

        pop_count = round(total * self.hard_negative_ratio)
        uniform_count = total - pop_count

        uniform_samples = torch.randint(
            self.num_items, (uniform_count,), device=target_device
        )
        pop_samples = torch.multinomial(
            self.popularity.to(target_device),
            pop_count,
            replacement=True,
        )
        samples = torch.cat([uniform_samples, pop_samples])

        if total > 1:
            samples = samples[torch.randperm(total, device=target_device)]
        return samples.reshape(shape)

    def _build_popularity(
        self, popularity: Optional[Union[torch.Tensor, Sequence[float]]]
    ) -> torch.Tensor:
        if popularity is None:
            weights = torch.ones(self.num_items, dtype=torch.float)
        else:
            weights = torch.as_tensor(popularity, dtype=torch.float)
            if weights.numel() != self.num_items:
                raise ValueError("popularity must contain one value per item.")
            weights = weights.reshape(self.num_items).clamp_min(0)

        total = weights.sum()
        if total <= 0:
            weights = torch.ones(self.num_items, dtype=torch.float)
            total = weights.sum()
        return weights / total

    @staticmethod
    def _normalize_shape(sample_shape: Union[int, Sequence[int]]) -> tuple[int, ...]:
        if isinstance(sample_shape, int):
            return (sample_shape,)
        return tuple(int(dim) for dim in sample_shape)
