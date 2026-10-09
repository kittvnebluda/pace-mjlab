# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Excitation signals for PACE data collection, normalized to [-1, 1] per sample."""

import torch
from torch import pi


def chirp(t: torch.Tensor, f0: float, f1: float, duration: float) -> torch.Tensor:
    """Linear chirp from f0 to f1 [Hz] over duration [s], shape of ``t``."""
    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t**2)
    return torch.sin(phase)


def steps(t: torch.Tensor, hold: float, scales: list[float]) -> torch.Tensor:
    """Steps held for ``hold`` [s]: the k-th level is (-1)^k * scales[k % len(scales)], shape of ``t``."""
    k = torch.floor(t / hold).to(torch.long)
    sign = 1.0 - 2.0 * (k % 2).to(t.dtype)
    return sign * torch.tensor(scales, dtype=t.dtype, device=t.device)[k % len(scales)]
