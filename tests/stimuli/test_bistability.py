"""Every number in App. C.10, checked against a waveform written out with numpy."""

import numpy as np
import pytest

from sonore_inference.stimuli import bistability as ab

FS = 20_000


def expected(interval, semitones):
    out = np.zeros(62_000)  # 3.1 s
    t = np.arange(1000) / FS  # 50 ms tones
    phase = (np.arange(200) + 0.5) / 200  # 10 ms raised-cosine ramps, sampled at midpoints
    gain = np.ones(1000)
    gain[:200] = (1 - np.cos(np.pi * phase)) / 2
    gain[-200:] = gain[:200][::-1]
    amplitude = np.sqrt(2) * 1e-6 * 10 ** (70 / 20)
    b_freq = 1000 * 2 ** (semitones / 12)
    for repeat in range(4):
        for slot, freq in enumerate([1000, b_freq, 1000]):  # A B A, then a silent slot
            onset = round((4 * repeat + slot) * interval * FS)
            out[onset : onset + 1000] += amplitude * np.cos(2 * np.pi * freq * t) * gain
    return out


def test_set_has_30_stimuli():
    stimuli = ab.bistability_set()
    assert len(stimuli) == 30
    assert {(c["semitones"], c["onset_interval"]) for c, _ in stimuli} == {
        (s, i) for s in (1, 3, 6, 9, 12) for i in (0.067, 0.083, 0.100, 0.117, 0.134, 0.150)
    }


@pytest.mark.parametrize("interval", [0.067, 0.083, 0.100, 0.117, 0.134, 0.150])
@pytest.mark.parametrize("semitones", [1, 12])
def test_waveform_matches_paper(interval, semitones):
    sound = ab.bistability_stimulus(interval, semitones)
    assert sound.fs == FS
    np.testing.assert_allclose(sound.data[:, 0], expected(interval, semitones), rtol=0, atol=1e-12)


def test_steady_level_is_70_db():
    sound = ab.bistability_stimulus(0.150, 6)
    steady = sound.data[200:800, 0]  # inside the first tone, past its ramp
    # 600 samples is 30 periods of 1000 Hz, so the RMS is exact
    assert 20 * np.log10(np.sqrt(np.mean(steady**2)) / 1e-6) == pytest.approx(70, abs=1e-9)
