"""The paper's sampling rate and level convention (App. C, first paragraph).

"All sounds were generated at 20 kHz. Stimulus levels for the model
experiments are specified in dB relative to an arbitrary model reference
value, 1e-6." We read the reference as an RMS value, so a sound at L dB has
RMS ``1e-6 * 10 ** (L / 20)``.
"""

import numpy as np
import sonore as so

FS = 20_000
REFERENCE_RMS = 1e-6


def rms_from_db(level_db: float) -> float:
    """RMS of a sound at ``level_db`` dB re ``REFERENCE_RMS``."""
    return REFERENCE_RMS * 10 ** (level_db / 20)


def sinusoid(duration: float, fs: float, freq: float, level_db: float, phase: float = 0.0) -> so.Sound:
    """``A cos(2 pi freq t + phase)`` with ``A = sqrt(2) * rms_from_db(level_db)``.

    The level is that of the unending sinusoid. ``sonore.pure_tone`` is not
    used because it scales its output to RMS 1 over its own duration, which
    differs slightly from a sinusoid's RMS when the duration is not a whole
    number of periods.
    """
    t = np.arange(round(duration * fs)) / fs
    return so.Sound(np.sqrt(2) * rms_from_db(level_db) * np.cos(2 * np.pi * freq * t + phase), fs)
