"""Differentiable renderers for the two source types the mistuned-harmonic
experiment needs: a harmonic tone and a whistle (a pure tone).

:func:`harmonic_tone` and :func:`whistle` are deliberately simpler than
BASS's sources [§2.1 of Cusimano et al., 2024]: frequency and levels are
constant over the event, and the event's onset, duration and ramps are
given. :func:`whistle_event` lets the onset and duration be inferred, and
:func:`trajectory_event` lets frequencies and levels change over time. Levels are in dB re RMS
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


def whistle_event(
    freq: torch.Tensor,
    level_db: torch.Tensor,
    onset: torch.Tensor,
    duration: torch.Tensor,
    *,
    fs: float,
    total_duration: float,
    ramp: float,
) -> torch.Tensor:
    """A whistle whose ``onset`` and ``duration`` [s] are tensors, so they can be inferred.

    The gate is the same raised-cosine ramp as :func:`ramp_gain`, written as a
    continuous function of time, so at an onset and duration of whole samples
    this equals :func:`whistle`; between samples it moves smoothly. The tone's
    phase starts at the onset, as in :func:`whistle`.
    """
    dtype, device = level_db.dtype, level_db.device
    n = torch.arange(round(total_duration * fs), dtype=dtype, device=device)
    midpoints = (n + 0.5) / fs  # where sonore samples its ramps
    offset = onset + duration

    def rise(seconds):
        return (1 - torch.cos(math.pi * (seconds / ramp).clamp(0, 1))) / 2

    gate = rise(midpoints - onset) * rise(offset - midpoints)
    amplitude = math.sqrt(2) * REFERENCE_RMS * 10 ** (level_db / 20)
    return amplitude * gate * torch.cos(2 * math.pi * freq * (n / fs - onset))


def _gate(midpoints: torch.Tensor, onset, duration, ramp: float) -> torch.Tensor:
    """The raised-cosine gate of :func:`whistle_event` at ``midpoints`` [s]."""

    def rise(seconds):
        return (1 - torch.cos(math.pi * (seconds / ramp).clamp(0, 1))) / 2

    return rise(midpoints - onset) * rise(onset + duration - midpoints)


def upsample(grid_values: torch.Tensor, grid_step: float, n_samples: int, fs: float) -> torch.Tensor:
    """Linear interpolation of values on a grid ``0, grid_step, 2 grid_step, ...`` [s]
    (last dimension) to the times ``n / fs`` of ``n_samples`` samples, held flat past the last point."""
    position = (torch.arange(n_samples, dtype=grid_values.dtype, device=grid_values.device) / fs) / grid_step
    position = position.clamp(max=grid_values.shape[-1] - 1)
    low = position.floor().long().clamp(max=grid_values.shape[-1] - 2)
    weight = position - low
    return grid_values[..., low] * (1 - weight) + grid_values[..., low + 1] * weight


def trajectory_event(
    freqs_hz: torch.Tensor,
    levels_db: torch.Tensor,
    onset,
    duration,
    *,
    grid_step: float,
    fs: float,
    total_duration: float,
    ramp: float,
) -> torch.Tensor:
    """Sinusoids whose frequencies and levels change over time, summed and gated as one event.

    ``freqs_hz`` and ``levels_db`` have shape ``(n_components, n_grid)``: each
    component's frequency [Hz] and level [dB] on a grid of times ``0,
    grid_step, ...`` covering the scene, interpolated linearly to the samples,
    as BASS samples its trajectories on a 10 ms grid [App. A.5 of Cusimano et
    al., 2024]. Each component's phase is the running integral of its
    frequency, zero at ``onset`` (cosine phase there), so with constant
    trajectories this is :func:`whistle_event` summed over components.
    ``onset`` and ``duration`` [s] may be tensors, to be inferred.
    """
    n_samples = round(total_duration * fs)
    freqs = upsample(freqs_hz, grid_step, n_samples, fs)
    levels = upsample(levels_db, grid_step, n_samples, fs)
    # integral of frequency up to each sample time n / fs (left Riemann sum)
    cycles = torch.nn.functional.pad(freqs.cumsum(-1)[..., :-1], (1, 0)) / fs
    # cycles at the onset, by linear interpolation, so the onset can move smoothly
    position = torch.as_tensor(onset, dtype=freqs.dtype) * fs
    low = position.detach().floor().long().clamp(0, n_samples - 2)
    weight = position - low
    cycles_at_onset = cycles[..., low] * (1 - weight) + cycles[..., low + 1] * weight
    amplitudes = math.sqrt(2) * REFERENCE_RMS * 10 ** (levels / 20)
    tones = amplitudes * torch.cos(2 * math.pi * (cycles - cycles_at_onset[..., None]))
    midpoints = (torch.arange(n_samples, dtype=freqs.dtype, device=freqs.device) + 0.5) / fs
    return tones.sum(0) * _gate(midpoints, onset, duration, ramp)
