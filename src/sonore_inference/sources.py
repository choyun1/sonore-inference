"""Differentiable renderers for the two source types the mistuned-harmonic
experiment needs: a harmonic tone and a whistle (a pure tone).

Both are deliberately simpler than BASS's sources [§2.1 of Cusimano et al.,
2024]: frequency and levels are constant over the event, and the event's
onset, duration and ramps are given, not inferred. Levels are in dB re RMS
1e-6 (each component's steady-state level), as in the stimuli.
"""

from __future__ import annotations

import math

import torch

REFERENCE_RMS = 1e-6


def ramp_gain(n_samples: int, n_ramp: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """Raised-cosine onset and offset ramps of ``n_ramp`` samples, sampled at
    segment midpoints as sonore's ``Sound.ramp`` does."""
    gain = torch.ones(n_samples, dtype=dtype, device=device)
    if n_ramp:
        phase = (torch.arange(n_ramp, dtype=dtype, device=device) + 0.5) / n_ramp
        ramp_up = (1 - torch.cos(math.pi * phase)) / 2
        gain[:n_ramp] = ramp_up
        gain[-n_ramp:] = ramp_up.flip(0)
    return gain


def _place(event: torch.Tensor, onset_samples: int, total_samples: int) -> torch.Tensor:
    """Put ``event`` (shape ``(..., n)``) at ``onset_samples`` in a silent signal of ``total_samples``."""
    after = total_samples - onset_samples - event.shape[-1]
    if onset_samples < 0 or after < 0:
        raise ValueError("the event does not fit in the signal")
    return torch.nn.functional.pad(event, (onset_samples, after))


def harmonic_tone(
    f0: torch.Tensor,
    levels_db: torch.Tensor,
    *,
    fs: float,
    onset: float,
    duration: float,
    total_duration: float,
    ramp: float,
    mistuning_hz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Harmonics ``1 .. len(levels_db)`` of ``f0`` [Hz] in cosine phase, with
    harmonic ``k`` at ``levels_db[k - 1]`` dB, from ``onset`` for ``duration``
    seconds inside ``total_duration``, with raised-cosine ramps of ``ramp`` s.
    ``mistuning_hz``, one value per harmonic, shifts each from ``k * f0``.
    """
    dtype, device = levels_db.dtype, levels_db.device
    n_event = round(duration * fs)
    t = torch.arange(n_event, dtype=dtype, device=device) / fs
    numbers = torch.arange(1, levels_db.shape[-1] + 1, dtype=dtype, device=device)
    freqs = numbers * f0
    if mistuning_hz is not None:
        freqs = freqs + mistuning_hz
    amplitudes = math.sqrt(2) * REFERENCE_RMS * 10 ** (levels_db / 20)
    event = (amplitudes[:, None] * torch.cos(2 * math.pi * freqs[:, None] * t)).sum(0)
    event = event * ramp_gain(n_event, round(ramp * fs), dtype=dtype, device=device)
    return _place(event, round(onset * fs), round(total_duration * fs))


def whistle(
    freq: torch.Tensor,
    level_db: torch.Tensor,
    *,
    fs: float,
    onset: float,
    duration: float,
    total_duration: float,
    ramp: float,
) -> torch.Tensor:
    """A pure tone at ``freq`` [Hz] and ``level_db``, placed and ramped as :func:`harmonic_tone`."""
    return harmonic_tone(
        freq,
        level_db.reshape(1),
        fs=fs,
        onset=onset,
        duration=duration,
        total_duration=total_duration,
        ramp=ramp,
    )
