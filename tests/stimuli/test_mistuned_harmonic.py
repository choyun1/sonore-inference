"""Every number in App. C.6, checked against a waveform written out with numpy."""

import numpy as np
import pytest

from sonore_inference.stimuli import mistuned_harmonic as mh

FS = 20_000


def raised_cosine_gain(n_total, n_ramp):
    # sonore samples the ramp at segment midpoints; the paper does not fix this detail
    phase = (np.arange(n_ramp) + 0.5) / n_ramp
    ramp_up = (1 - np.cos(np.pi * phase)) / 2
    gain = np.ones(n_total)
    gain[:n_ramp], gain[-n_ramp:] = ramp_up, ramp_up[::-1]
    return gain


def expected(f0, number, percent):
    harmonics = range(1, 11) if f0 == 400 else range(1, 13)
    t = np.arange(8000) / FS  # 400 ms
    amplitude = np.sqrt(2) * 1e-6 * 10 ** (60 / 20)  # each component at 60 dB re 1e-6 RMS
    tone = sum(
        amplitude * np.cos(2 * np.pi * (n * f0 + (f0 * percent / 100 if n == number else 0)) * t)
        for n in harmonics
    )
    tone *= raised_cosine_gain(8000, 200)  # 10 ms ramps
    return np.concatenate([np.zeros(1000), tone, np.zeros(1000)])  # 50 ms each side


def test_set_has_63_stimuli_with_every_condition():
    stimuli = mh.mistuned_harmonic_set()
    assert len(stimuli) == 63
    conditions = {tuple(c.values()) for c, _ in stimuli}
    assert conditions == {
        (f0, n, p) for f0 in (100, 200, 400) for n in (1, 2, 3) for p in (0, 5, 10, 20, 30, 40, 50)
    }


@pytest.mark.parametrize("f0", [100.0, 200.0, 400.0])
@pytest.mark.parametrize("number", [1, 2, 3])
@pytest.mark.parametrize("percent", [0, 5, 50])
def test_waveform_matches_paper(f0, number, percent):
    sound = mh.mistuned_harmonic_stimulus(f0, number, percent)
    assert sound.fs == FS
    np.testing.assert_allclose(sound.data[:, 0], expected(f0, number, percent), rtol=0, atol=1e-12)


def test_component_levels_from_spectrum():
    # 200 ms in the steady part: every component falls on a 5 Hz DFT bin
    sound = mh.mistuned_harmonic_stimulus(200.0, 2, 20)
    segment = sound.data[3000:7000, 0]
    spectrum = 2 * np.abs(np.fft.rfft(segment)) / len(segment)
    freqs = np.fft.rfftfreq(len(segment), 1 / FS)
    present = spectrum > 1e-9
    expected_freqs = [200 * n for n in range(1, 13) if n != 2] + [440]
    assert sorted(freqs[present]) == sorted(expected_freqs)
    levels_db = 20 * np.log10(spectrum[present] / np.sqrt(2) / 1e-6)
    np.testing.assert_allclose(levels_db, 60, atol=1e-9)


def test_relative_to_harmonic():
    sound = mh.mistuned_harmonic_stimulus(100.0, 3, 10, relative_to="harmonic")
    segment = sound.data[3000:7000, 0]
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1 / FS)
    assert 330 in freqs[spectrum > 1e-9 * spectrum.max()]
