"""Every number in App. C.7 that can be checked without re-running the Klatt synthesizer."""

import numpy as np
import pytest
import sonore as so

from sonore_inference.stimuli import asynchronous_onsets as ao

FS = 20_000
VOWEL_ONSET = 1000 + 4800  # 50 ms of silence, then the 240 ms lead
VOWEL_SAMPLES = 1200  # 60 ms


def db(rms):
    return 20 * np.log10(rms / 1e-6)


def test_set_has_28_stimuli():
    stimuli = ao.asynchronous_onsets_set()
    assert len(stimuli) == 28
    assert {(c["continuum"], c["f1"]) for c, _ in stimuli} == {
        (c, f1)
        for c in ("basic", "shifted", "early_32ms", "early_240ms")
        for f1 in (375, 396, 417, 438, 459, 480, 500)
    }
    lengths = {s.n_samples for _, s in stimuli}
    assert lengths == {1000 + 4800 + 1200 + 1000}  # all the same length


def test_vowel_uses_the_papers_klatt_parameters():
    reference = so.klatt_synthesize(
        0.060,
        FS,
        F0=125,
        SS=1,
        F1=438,
        B1=70,
        F2=2300,
        B2=70,
        F3=2900,
        B3=150,
        F4=3800,
        B4=200,
        F5=4600,
        B5=200,
    )
    phase = (np.arange(320) + 0.5) / 320  # 16 ms linear ramps, sampled at midpoints
    gain = np.ones(VOWEL_SAMPLES)
    gain[:320], gain[-320:] = phase, phase[::-1]
    reference = reference.data[:, 0] * gain
    reference *= 1e-6 * 10 ** (56 / 20) / np.sqrt(np.mean(reference**2))  # 56 dB overall
    np.testing.assert_allclose(ao.vowel(438.0).data[:, 0], reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("f1", [375.0, 500.0])
def test_basic_continuum_is_the_vowel_alone(f1):
    sound = ao.asynchronous_onsets_stimulus("basic", f1).data[:, 0]
    assert not sound[:VOWEL_ONSET].any() and not sound[VOWEL_ONSET + VOWEL_SAMPLES :].any()
    assert db(np.sqrt(np.mean(sound[VOWEL_ONSET : VOWEL_ONSET + VOWEL_SAMPLES] ** 2))) == pytest.approx(56)


@pytest.mark.parametrize(
    "continuum, lead_samples", [("shifted", 0), ("early_32ms", 640), ("early_240ms", 4800)]
)
@pytest.mark.parametrize("f1", [375.0, 438.0, 500.0])
def test_tone_timing_level_and_phase(continuum, lead_samples, f1):
    sound = ao.asynchronous_onsets_stimulus(continuum, f1)
    vowel = ao.asynchronous_onsets_stimulus("basic", f1)
    tone = sound.data[:, 0] - vowel.data[:, 0]
    nonzero = np.flatnonzero(tone)
    assert nonzero[0] == VOWEL_ONSET - lead_samples  # starts early by the lead
    assert nonzero[-1] == VOWEL_ONSET + VOWEL_SAMPLES - 1  # ends with the vowel
    # 16 ms linear ramps: the first sample has 1/640 of the steady amplitude
    steady_amplitude = np.abs(tone[nonzero[0] + 320 : nonzero[-1] - 320]).max()
    assert abs(tone[nonzero[0]]) <= steady_amplitude / 320

    # measure the 500 Hz component in the vowel alone and in vowel + tone
    window = (VOWEL_ONSET / FS + 0.020, VOWEL_ONSET / FS + 0.044)
    before = ao.harmonic_component(vowel, 125, 4, window)
    after = ao.harmonic_component(sound, 125, 4, window)
    tone_alone = after - before
    assert 20 * np.log10(abs(tone_alone) / abs(before)) == pytest.approx(6.0, abs=1e-6)  # 6 dB above
    # in phase, the sum is 20 log10(1 + 10 ** (6 / 20)) = 9.53 dB up: the paper's 9.5 dB
    assert 20 * np.log10(abs(after) / abs(before)) == pytest.approx(9.53, abs=0.005)
