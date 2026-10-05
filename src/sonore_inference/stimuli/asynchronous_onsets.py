"""Asynchronous onsets (App. C.7, after Darwin & Sutherland, 1984, Exp. 1).

From the paper: Klatt (1980) vowels, 60 ms long, f0 125 Hz, overall level
56 dB; F1 of 375, 396, 417, 438, 459, 480 or 500 Hz with bandwidth 70 Hz;
F2-F5 at 2300, 2900, 3800 and 4600 Hz with bandwidths 70, 150, 200 and
200 Hz; 16 ms linear onset and offset ramps. Four continua: the basic vowels;
"shifted", with a 500 Hz tone of the same onset and offset as the vowel; and
two "early onset" continua whose tone starts 32 or 240 ms before the vowel.
Each tone is 6 dB above the vowel's 500 Hz component and adds to it
constructively, raising the level at 500 Hz by 9.5 dB; it has 16 ms linear
ramps. All 28 stimuli have the same duration, the vowel starts at the same
time in each, and the one with the largest asynchrony has 50 ms of silence.

Assumed, because the text does not say:

- Every tone ends when its vowel ends (as in Darwin & Sutherland).
- The 500 Hz component is measured on the vowel at its final level, by a
  least-squares fit of all harmonics of 125 Hz between 20 and 44 ms (after
  the onset ramp and the formant filters' onset transient, before the offset
  ramp). The tone is 6 dB above it and in phase with it, so the sum is
  20 log10(1 + 10 ** (6 / 20)) = 9.53 dB above the component, which matches
  the paper's 9.5 dB.
- "56 dB" is the vowel's RMS over its 60 ms, ramps included.
- The tone's level is steady-state, before its ramps.
- The 50 ms of silence goes both before and after the longest stimulus, as
  for the mistuned harmonic.
- Klatt's voicing source is the 1980 impulse source (sonore's ``SS=1``).
"""

import numpy as np
import sonore as so

from sonore_inference.stimuli.levels import FS, rms_from_db

VOWEL_DURATION = 0.060
VOWEL_F0 = 125.0
VOWEL_LEVEL_DB = 56.0
F1S = (375.0, 396.0, 417.0, 438.0, 459.0, 480.0, 500.0)
VOWEL_FORMANTS = {
    "B1": 70.0,
    "F2": 2300.0,
    "B2": 70.0,
    "F3": 2900.0,
    "B3": 150.0,
    "F4": 3800.0,
    "B4": 200.0,
    "F5": 4600.0,
    "B5": 200.0,
}
RAMP = 0.016
TONE_FREQ = 500.0
TONE_GAIN_DB = 6.0
# continuum name: how long the tone starts before the vowel (None = no tone)
CONTINUA = {"basic": None, "shifted": 0.0, "early_32ms": 0.032, "early_240ms": 0.240}
PADDING = 0.050
FIT_WINDOW = (0.020, 0.044)


def vowel(f1: float, fs: float = FS) -> so.Sound:
    """One vowel of the basic continuum, ramped and at 56 dB."""
    sound = so.klatt_synthesize(VOWEL_DURATION, fs, F0=VOWEL_F0, F1=f1, SS=1, **VOWEL_FORMANTS)
    return sound.ramp(RAMP, shape="linear").normalize(rms_from_db(VOWEL_LEVEL_DB))


def harmonic_component(sound: so.Sound, f0: float, number: int, window: tuple[float, float]) -> complex:
    """Complex amplitude ``a e^{i phi}`` of harmonic ``number`` of ``f0`` in ``sound``.

    Fits ``sum_n Re(c_n e^{i 2 pi n f0 t})`` over every harmonic below
    Nyquist to the samples in ``window`` (seconds) by least squares, with
    ``t`` measured from the sound's first sample.
    """
    fs = sound.fs
    start, stop = round(window[0] * fs), round(window[1] * fs)
    t = np.arange(start, stop) / fs
    numbers = np.arange(1, int(np.ceil(fs / 2 / f0)))
    phase = 2 * np.pi * np.outer(t, numbers * f0)
    design = np.hstack([np.cos(phase), -np.sin(phase)])
    coefs, *_ = np.linalg.lstsq(design, sound.data[start:stop, 0], rcond=None)
    index = number - 1
    return complex(coefs[index], coefs[index + len(numbers)])


def matched_tone(vowel_sound: so.Sound, lead: float) -> so.Sound:
    """The 500 Hz tone for ``vowel_sound``, starting ``lead`` s before it and ending with it.

    Its phase is set from the vowel's time axis, so that the two are in phase
    where they overlap.
    """
    fs = vowel_sound.fs
    component = harmonic_component(vowel_sound, VOWEL_F0, round(TONE_FREQ / VOWEL_F0), FIT_WINDOW)
    amplitude = abs(component) * 10 ** (TONE_GAIN_DB / 20)
    lead_samples = round(lead * fs)
    t = (np.arange(lead_samples + vowel_sound.n_samples) - lead_samples) / fs
    tone = amplitude * np.cos(2 * np.pi * TONE_FREQ * t + np.angle(component))
    return so.Sound(tone, fs).ramp(RAMP, shape="linear")


def asynchronous_onsets_stimulus(
    continuum: str,
    f1: float,
    *,
    padding: tuple[float, float] = (PADDING, PADDING),
    fs: float = FS,
) -> so.Sound:
    """One stimulus: the vowel with F1 = ``f1``, plus the tone for ``continuum``.

    Every stimulus has the length of the longest (240 ms lead) plus
    ``padding`` (before, after), with the vowel starting at the same time.
    """
    if continuum not in CONTINUA:
        raise ValueError(f"continuum must be one of {sorted(CONTINUA)}, got {continuum!r}")
    max_lead_samples = round(max(lead for lead in CONTINUA.values() if lead is not None) * fs)
    before, after = round(padding[0] * fs), round(padding[1] * fs)
    vowel_sound = vowel(f1, fs)
    vowel_onset = before + max_lead_samples
    total = vowel_onset + vowel_sound.n_samples + after
    parts = [vowel_sound.pad_to(vowel_onset + vowel_sound.n_samples, align="end")]
    lead = CONTINUA[continuum]
    if lead is not None:
        tone = matched_tone(vowel_sound, lead)
        parts.append(tone.pad_to(vowel_onset + vowel_sound.n_samples, align="end"))
    return so.mix(parts).pad_to(total)


def asynchronous_onsets_set(**kwargs) -> list[tuple[dict, so.Sound]]:
    """All 28 stimuli as (condition, sound) pairs; ``kwargs`` go to the stimulus function."""
    return [
        ({"continuum": continuum, "f1": f1}, asynchronous_onsets_stimulus(continuum, f1, **kwargs))
        for continuum in CONTINUA
        for f1 in F1S
    ]
