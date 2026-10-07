"""Two simultaneous notes (milestone (b), design D3 (b‴) and docs/design/milestone-b.md, B1).

Not from the paper: these stimuli test whether the model hears two harmonic
notes sounding together as two sources. Four intervals, graded from easy to
hard, each with the upper note starting 0, 10, 20, 40 or 80 ms after the
lower one:

- ``tritone_different_spectra``: 200 Hz and 200 * sqrt(2) = 282.8 Hz; the
  lower note's harmonics all at 60 dB, the upper note's falling 6 dB per
  octave of harmonic number (60 - 6 log2 n dB);
- ``tritone``: the same with both spectra flat;
- ``just_fifth``: 200 and 300 Hz, flat;
- ``octave``: 200 and 400 Hz, flat.

Each note has every harmonic up to 2400 Hz (12, 8, 8 and 6 harmonics for
200, 282.8, 300 and 400 Hz), so at the octave every upper component falls on
a lower one. Where components coincide (the fifth's even upper harmonics, all
of the octave's), they add in phase, 6 dB above either alone, at every
asynchrony: each asynchrony is a whole number of their periods. Levels are
steady-state per component; components start in cosine phase at their note's
onset; ramps are 10 ms raised cosine; 50 ms of silence before and after, as
for the mistuned harmonic. Four controls play one note alone (200, 282.8, 300
or 400 Hz, flat), timed as the lower note.

Assumed (the design gives the note length and that both end together):

- The lower note lasts 400 ms; the upper note starts ``asynchrony`` later
  and ends with it, so it lasts 400 ms minus the asynchrony. Every stimulus
  is then 500 ms long.
"""

import numpy as np
import sonore as so

from sonore_inference.stimuli.levels import FS, sinusoid

DURATION = 0.400
LOWER_F0 = 200.0
UPPER_F0S = {
    "tritone_different_spectra": 200.0 * np.sqrt(2),
    "tritone": 200.0 * np.sqrt(2),
    "just_fifth": 300.0,
    "octave": 400.0,
}
DIFFERENT_SPECTRA = ("tritone_different_spectra",)
ASYNCHRONIES = (0.0, 0.010, 0.020, 0.040, 0.080)
CONTROL_F0S = (200.0, 200.0 * np.sqrt(2), 300.0, 400.0)
MAX_FREQ = 2400.0
COMPONENT_LEVEL_DB = 60.0
FALLING_DB_PER_OCTAVE = 6.0
RAMP = 0.010
PADDING = 0.050


def harmonic_numbers(f0: float) -> np.ndarray:
    """Every harmonic up to ``MAX_FREQ`` (inclusive, to within rounding)."""
    return np.arange(1, int(np.floor(MAX_FREQ / f0 + 1e-9)) + 1)


def harmonic_levels_db(f0: float, falling: bool = False) -> np.ndarray:
    """Each harmonic's level: flat at 60 dB, or falling 6 dB per octave of harmonic number."""
    numbers = harmonic_numbers(f0)
    if not falling:
        return np.full(len(numbers), COMPONENT_LEVEL_DB)
    return COMPONENT_LEVEL_DB - FALLING_DB_PER_OCTAVE * np.log2(numbers)


def unshared_upper_harmonics(interval: str) -> np.ndarray:
    """Harmonic numbers of the upper note that do not coincide with a harmonic of the lower note.

    These are the components hypothesis B2 (B) explains as whistles: all 8 for
    the tritone, 1, 3, 5 and 7 for the just fifth, none for the octave.
    """
    upper = UPPER_F0S[interval]
    numbers = harmonic_numbers(upper)
    ratios = numbers * upper / LOWER_F0
    return numbers[np.abs(ratios - np.round(ratios)) > 1e-6]


def note(f0: float, duration: float, *, falling: bool = False, fs: float = FS) -> so.Sound:
    """One ramped harmonic note, cosine phase at its onset, no padding."""
    components = [
        sinusoid(duration, fs, number * f0, level)
        for number, level in zip(harmonic_numbers(f0), harmonic_levels_db(f0, falling), strict=True)
    ]
    return so.mix(components).ramp(RAMP, shape="cosine")


def two_notes_stimulus(interval: str, asynchrony: float, *, fs: float = FS) -> so.Sound:
    """The lower note and the ``interval``'s upper note, which starts ``asynchrony`` [s] later."""
    if interval not in UPPER_F0S:
        raise ValueError(f"interval must be one of {list(UPPER_F0S)}, got {interval!r}")
    if not 0 <= asynchrony < DURATION - 2 * RAMP:
        raise ValueError(f"asynchrony must be in [0, {DURATION - 2 * RAMP}) s, got {asynchrony}")
    lower = note(LOWER_F0, DURATION, fs=fs).pad(PADDING, PADDING)
    upper_duration = round((DURATION - asynchrony) * fs) / fs
    upper = note(UPPER_F0S[interval], upper_duration, falling=interval in DIFFERENT_SPECTRA, fs=fs)
    upper = upper.pad(PADDING + asynchrony, PADDING)
    return so.mix([lower, upper])


def single_note_control(f0: float, *, fs: float = FS) -> so.Sound:
    """One flat note alone, timed as the lower note of the two-note stimuli."""
    return note(f0, DURATION, fs=fs).pad(PADDING, PADDING)


def two_notes_set(**kwargs) -> list[tuple[dict, so.Sound]]:
    """The 20 two-note stimuli and the 4 controls as (condition, sound) pairs.

    A control's condition has ``interval`` ``"control"``, its ``f0`` and no asynchrony.
    """
    stimuli = []
    for interval in UPPER_F0S:
        for asynchrony in ASYNCHRONIES:
            condition = {"interval": interval, "asynchrony": asynchrony}
            stimuli.append((condition, two_notes_stimulus(interval, asynchrony, **kwargs)))
    for f0 in CONTROL_F0S:
        stimuli.append(({"interval": "control", "f0": float(f0)}, single_note_control(f0, **kwargs)))
    return stimuli
