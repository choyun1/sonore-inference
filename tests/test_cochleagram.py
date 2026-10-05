"""The cochleagram against sonore (filters and envelopes, float64) and against closed forms."""

import numpy as np
import pytest
import sonore as so
import torch

from sonore_inference.cochleagram import (
    Cochleagram,
    FFTCochleagram,
    freq_to_erb,
    gammatone_response,
    gaussian_log_likelihood,
)

FS = 20_000


def sonore_bank(cochleagram):
    """sonore's bare gammatone bank with the same centers and bandwidths.
    sonore's knots include one point beyond each end, where its edge filters
    would take over; with edges=False those points only fix the spacing."""
    erbs = np.linspace(freq_to_erb(cochleagram.f_lo), freq_to_erb(cochleagram.f_hi), cochleagram.n_channels)
    step = erbs[1] - erbs[0]
    knots = np.concatenate([[erbs[0] - step], erbs, [erbs[-1] + step]])
    return so.gammatone_filterbank(
        centers=so.erb_to_freq(knots),
        order=cochleagram.order,
        bandwidth_factor=cochleagram.bandwidth_factor,
        edges=False,
    )


def test_centers_match_bass():
    cfs = Cochleagram().cfs
    assert len(cfs) == 64
    assert cfs[0] == pytest.approx(20.0) and cfs[-1] == pytest.approx(9423.0)
    np.testing.assert_allclose(np.diff(freq_to_erb(cfs)), np.diff(freq_to_erb(cfs))[0])


@pytest.mark.parametrize("n", [4000, 4001])
def test_filters_match_sonore(n):
    cochleagram = Cochleagram()
    bank = sonore_bank(cochleagram)
    np.testing.assert_allclose(bank.band_cfs, cochleagram.cfs, rtol=1e-12)
    ours = gammatone_response(
        np.fft.rfftfreq(n, 1 / FS), cochleagram.cfs, cochleagram.order, cochleagram.bandwidth_factor
    )
    np.testing.assert_allclose(ours, bank.response(np.fft.rfftfreq(n, 1 / FS)), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [3000, 3001])
def test_envelopes_match_sonore(n):
    cochleagram = Cochleagram(n_channels=16, pad=2000)
    rng = np.random.default_rng(0)
    sound = so.Sound(rng.standard_normal(n), FS)
    expected = sonore_bank(cochleagram).analyze(sound, pad=2000 / FS).envelopes().data[:, :, 0].T
    ours = cochleagram.envelopes(torch.tensor(sound.data[:, 0])).numpy()
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-10 * expected.max())


def test_tone_at_a_center_reads_its_level():
    cochleagram = Cochleagram()
    channel = 40
    tone = so.pure_tone(0.5, FS, cochleagram.cfs[channel]).data[:, 0] * 1e-6 * 10 ** (60 / 20)
    levels = cochleagram(torch.as_tensor(tone)).numpy()
    times = cochleagram.frame_times(len(tone))
    steady = (times > 0.1) & (times < 0.4)
    np.testing.assert_allclose(levels[channel, steady], 60, atol=0.01)
    # channels far from the tone are at the floor
    assert levels[0, steady].max() == 20.0


def test_frames():
    cochleagram = Cochleagram()
    assert (cochleagram.frame_samples, cochleagram.hop_samples) == (500, 200)
    levels = cochleagram(torch.zeros(2, 10_000))
    assert levels.shape == (2, 64, (10_000 - 500) // 200 + 1)
    assert (levels == 20).all()  # silence sits at the floor


def test_gradients():
    cochleagram = Cochleagram(n_channels=4, f_lo=200, f_hi=2000, pad=300, floor_db=-1000)
    waveform = (
        1e-3 * torch.randn(900, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: cochleagram(x).sum(), (waveform,))


def test_gaussian_log_likelihood():
    observed = torch.tensor([[[30.0, 40.0]]], dtype=torch.float64)
    predicted = torch.tensor([[[35.0, 40.0]]], dtype=torch.float64)
    sigma = 10.0
    expected = sum(
        -0.5 * ((o - p) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)) for o, p in [(30, 35), (40, 40)]
    )
    assert gaussian_log_likelihood(observed, predicted, sigma).item() == pytest.approx(expected)


def test_fft_cochleagram_reads_a_tone_at_a_center():
    cochleagram = FFTCochleagram()
    channel = 30
    tone = so.pure_tone(0.5, FS, cochleagram.cfs[channel]).data[:, 0] * 1e-6 * 10 ** (60 / 20)
    levels = cochleagram(torch.as_tensor(tone)).numpy()
    assert levels.shape == Cochleagram()(torch.as_tensor(tone)).shape
    times = cochleagram.frame_times(len(tone))
    steady = (times > 0.1) & (times < 0.4)
    np.testing.assert_allclose(levels[channel, steady], 60, atol=0.01)
    assert cochleagram.fft_size == 1024  # Ellis: next power of two above twice the 500-sample window


def test_fft_channels_are_wider_at_low_frequencies():
    # a tone 40 Hz above a 104 Hz channel: far down the half-ERB gammatone,
    # still inside the 25 ms window's main lobe
    channel = int(np.argmin(np.abs(Cochleagram().cfs - 104)))
    cf = Cochleagram().cfs[channel]
    tone = torch.as_tensor(so.pure_tone(0.5, FS, cf + 40).data[:, 0] * 1e-6 * 10 ** (60 / 20))
    gammatone = Cochleagram()(tone)[channel, 10:40].mean()
    fft = FFTCochleagram()(tone)[channel, 10:40].mean()
    assert fft - gammatone > 10


def test_fft_cochleagram_gradients():
    cochleagram = FFTCochleagram(n_channels=4, f_lo=200, f_hi=2000, floor_db=-1000, frame=0.005, hop=0.0025)
    waveform = (
        1e-3 * torch.randn(300, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    ).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: cochleagram(x).sum(), (waveform,))


def test_bass_gain_is_a_fixed_offset_per_channel_and_largest_at_low_frequencies():
    calibrated, bass = FFTCochleagram(floor_db=-1000), FFTCochleagram(floor_db=-1000, bass_gain=True)
    noise = torch.randn(10_000, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    tone = torch.as_tensor(so.pure_tone(0.5, FS, 1000.0).data[:, 0] * 1e-3)
    offsets = [(calibrated(x) - bass(x))[:, 5:40] for x in (1e-3 * noise, tone)]
    np.testing.assert_allclose(offsets[0], offsets[0][:, :1].expand_as(offsets[0]), atol=1e-9)
    np.testing.assert_allclose(offsets[0], offsets[1], atol=1e-9)
    per_channel = offsets[0][:, 0].numpy()
    low, high = np.argmin(np.abs(bass.cfs - 104)), np.argmin(np.abs(bass.cfs - 3165))
    # BASS's own gammatonegram, run on 60 dB tones, read about 47 dB at 104 Hz and 56 dB at 3165 Hz
    np.testing.assert_allclose(per_channel[[low, high]], [12.9, 3.7], atol=0.5)
