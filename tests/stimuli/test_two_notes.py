"""The two-note stimuli of milestone (b), checked against waveforms written out with numpy."""

import numpy as np
import pytest

from sonore_inference.stimuli import two_notes as tn

FS = 20_000
AMPLITUDE_60_DB = np.sqrt(2) * 1e-6 * 10 ** (60 / 20)


def raised_cosine_gain(n_total, n_ramp):
    phase = (np.arange(n_ramp) + 0.5) / n_ramp
    ramp_up = (1 - np.cos(np.pi * phase)) / 2
    gain = np.ones(n_total)
    gain[:n_ramp], gain[-n_ramp:] = ramp_up, ramp_up[::-1]
    return gain


def expected_note(f0, n_harmonics, start, n_samples, falling=False):
    t = np.arange(n_samples) / FS
    tone = sum(
        AMPLITUDE_60_DB * 10 ** ((-6 * np.log2(n) if falling else 0) / 20) * np.cos(2 * np.pi * n * f0 * t)
        for n in range(1, n_harmonics + 1)
    )
    out = np.zeros(10_000)  # 500 ms
    out[start : start + n_samples] = tone * raised_cosine_gain(n_samples, 200)
    return out


@pytest.mark.parametrize(
    "interval, upper_f0, n_upper, falling",
    [
        ("tritone_different_spectra", 200 * np.sqrt(2), 8, True),
        ("tritone", 200 * np.sqrt(2), 8, False),
        ("just_fifth", 300.0, 8, False),
        ("octave", 400.0, 6, False),
    ],
)
@pytest.mark.parametrize("asynchrony_ms", [0, 10, 80])
def test_waveform(interval, upper_f0, n_upper, falling, asynchrony_ms):
    sound = tn.two_notes_stimulus(interval, asynchrony_ms / 1000)
    assert sound.fs == FS
    lag = asynchrony_ms * 20  # samples
    expected = expected_note(200.0, 12, 1000, 8000) + expected_note(
        upper_f0, n_upper, 1000 + lag, 8000 - lag, falling
    )
    np.testing.assert_allclose(sound.data[:, 0], expected, rtol=0, atol=1e-12)


def test_set_has_20_stimuli_and_4_controls():
    stimuli = tn.two_notes_set()
    assert len(stimuli) == 24
    pairs = {(c["interval"], c["asynchrony"]) for c, _ in stimuli if c["interval"] != "control"}
    assert len(pairs) == 20
    controls = [c["f0"] for c, _ in stimuli if c["interval"] == "control"]
    assert controls == pytest.approx([200.0, 282.8427, 300.0, 400.0])
    assert all(len(sound.data) == 10_000 for _, sound in stimuli)


def test_control_is_the_lower_note_alone():
    np.testing.assert_allclose(
        tn.single_note_control(200.0).data[:, 0], expected_note(200.0, 12, 1000, 8000), rtol=0, atol=1e-12
    )


def test_harmonics_stop_at_2400_hz():
    assert [len(tn.harmonic_numbers(f0)) for f0 in tn.CONTROL_F0S] == [12, 8, 8, 6]


def test_unshared_upper_harmonics():
    assert list(tn.unshared_upper_harmonics("tritone")) == list(range(1, 9))
    assert list(tn.unshared_upper_harmonics("just_fifth")) == [1, 3, 5, 7]
    assert list(tn.unshared_upper_harmonics("octave")) == []


def test_bad_arguments():
    with pytest.raises(ValueError):
        tn.two_notes_stimulus("major_third", 0.0)
    with pytest.raises(ValueError):
        tn.two_notes_stimulus("octave", 0.39)
