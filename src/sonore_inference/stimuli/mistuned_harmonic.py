"""Mistuned harmonic (App. C.6, after Moore, Glasberg & Peters, 1986).

From the paper: 400 ms complex tones with f0 of 100, 200 or 400 Hz; equal
amplitude harmonics, each at 60 dB; harmonics 1-12 (1-10 at 400 Hz); 10 ms
raised-cosine onset and offset ramps; harmonic 1, 2 or 3 mistuned by 0, 5,
10, 20, 30, 40 or 50% of the fundamental frequency; 63 stimuli; each padded
with 50 ms of silence.

Assumed, because the text does not say:

- Mistuning is a percentage of f0, as App. C.6 and Fig. 7 have it; §3.1.2.2
  says "of the harmonic frequency" instead (``relative_to="harmonic"``).
- Mistuning is upward.
- Components start in cosine phase.
- Levels are steady-state, before ramps.
- The 50 ms of silence goes both before and after the tone.
"""

import numpy as np
import sonore as so

from sonore_inference.stimuli.levels import FS, sinusoid

DURATION = 0.400
F0S = (100.0, 200.0, 400.0)
MISTUNED_HARMONICS = (1, 2, 3)
MISTUNING_PERCENTS = (0, 5, 10, 20, 30, 40, 50)
COMPONENT_LEVEL_DB = 60.0
RAMP = 0.010
PADDING = 0.050


def harmonic_numbers(f0: float) -> np.ndarray:
    """Harmonics 1-12, or 1-10 at f0 = 400 Hz."""
    return np.arange(1, 11 if f0 == 400 else 13)


def mistuned_harmonic_stimulus(
    f0: float,
    mistuned_harmonic: int,
    mistuning_percent: float,
    *,
    relative_to: str = "f0",
    phases: str = "cosine",
    padding: tuple[float, float] = (PADDING, PADDING),
    fs: float = FS,
) -> so.Sound:
    """One complex tone with ``mistuned_harmonic`` shifted up by ``mistuning_percent``.

    ``relative_to`` is ``"f0"`` (the shift is a percentage of f0) or
    ``"harmonic"`` (a percentage of the harmonic's own frequency).
    ``phases`` is ``"cosine"`` or ``"sine"``. ``padding`` is the silence
    (before, after) in seconds.
    """
    if relative_to not in ("f0", "harmonic"):
        raise ValueError(f"relative_to must be 'f0' or 'harmonic', got {relative_to!r}")
    start_phase = {"cosine": 0.0, "sine": -np.pi / 2}[phases]
    components = []
    for number in harmonic_numbers(f0):
        freq = number * f0
        if number == mistuned_harmonic:
            base = f0 if relative_to == "f0" else freq
            freq += base * mistuning_percent / 100
        components.append(sinusoid(DURATION, fs, freq, COMPONENT_LEVEL_DB, start_phase))
    tone = so.mix(components).ramp(RAMP, shape="cosine")
    return tone.pad(*padding)


def mistuned_harmonic_set(**kwargs) -> list[tuple[dict, so.Sound]]:
    """All 63 stimuli as (condition, sound) pairs; ``kwargs`` go to the stimulus function."""
    stimuli = []
    for f0 in F0S:
        for number in MISTUNED_HARMONICS:
            for percent in MISTUNING_PERCENTS:
                condition = {"f0": f0, "mistuned_harmonic": number, "mistuning_percent": percent}
                stimuli.append((condition, mistuned_harmonic_stimulus(f0, number, percent, **kwargs)))
    return stimuli
