"""ABA bistability (App. C.10, after Bregman & Ahad, 1996, and van Noorden, 1975).

From the paper: three tones (A B A) followed by a silence as long as one
onset-to-onset interval; onset-to-onset intervals of 67, 83, 100, 117, 134 or
150 ms; A at 1000 Hz and B 1, 3, 6, 9 or 12 semitones higher; tones at 70 dB,
50 ms long, with 10 ms raised-cosine ramps; the triplet repeated four times;
every stimulus padded with silence to 3.1 s; 30 stimuli.

Assumed, because the text does not say:

- The first tone starts at time 0 and all the padding comes after.
- Each tone starts in cosine phase.
- "70 dB" is each tone's steady-state level, before ramps.
"""

import sonore as so

from sonore_inference.stimuli.levels import FS, sinusoid

ONSET_INTERVALS = (0.067, 0.083, 0.100, 0.117, 0.134, 0.150)
SEMITONES = (1, 3, 6, 9, 12)
A_FREQ = 1000.0
TONE_DURATION = 0.050
TONE_LEVEL_DB = 70.0
RAMP = 0.010
REPEATS = 4
TOTAL_DURATION = 3.1


def tone_onsets(onset_interval: float, repeats: int = REPEATS) -> list[tuple[float, str]]:
    """(onset time, "A" or "B") for every tone; each triplet takes four intervals."""
    onsets = []
    for repeat in range(repeats):
        start = 4 * repeat * onset_interval
        onsets += [(start, "A"), (start + onset_interval, "B"), (start + 2 * onset_interval, "A")]
    return onsets


def bistability_stimulus(
    onset_interval: float,
    semitones: float,
    *,
    repeats: int = REPEATS,
    total_duration: float = TOTAL_DURATION,
    fs: float = FS,
) -> so.Sound:
    """One ABA sequence, B ``semitones`` above A, padded to ``total_duration``."""
    freqs = {"A": A_FREQ, "B": A_FREQ * 2 ** (semitones / 12)}
    total_samples = round(total_duration * fs)
    tones = []
    for onset, name in tone_onsets(onset_interval, repeats):
        tone = sinusoid(TONE_DURATION, fs, freqs[name], TONE_LEVEL_DB).ramp(RAMP, shape="cosine")
        tones.append(tone.pad_to(tone.n_samples + round(onset * fs), align="end"))
    sequence = so.mix(tones)
    return sequence.pad_to(total_samples)


def bistability_set(**kwargs) -> list[tuple[dict, so.Sound]]:
    """All 30 stimuli as (condition, sound) pairs; ``kwargs`` go to the stimulus function."""
    return [
        (
            {"semitones": semitones, "onset_interval": interval},
            bistability_stimulus(interval, semitones, **kwargs),
        )
        for semitones in SEMITONES
        for interval in ONSET_INTERVALS
    ]
