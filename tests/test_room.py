"""The room's expected band energy against sonore's synth_ir
(docs/design/source-reverb.md, R2-R3)."""

import numpy as np
import pytest
import sonore as so
import torch
from sonore.spatial.reverb import _model

from sonore_inference.room import RoomGain, band_rt60s, cosine_responses, onset_levels_db

FS = 20_000.0


@pytest.mark.parametrize(("n_bands", "f_lo", "f_hi"), [(39, 20.0, 4000.0), (32, 20.0, 10000.0)])
def test_cosine_responses_match_sonore(n_bands, f_lo, f_hi):
    freqs = np.concatenate([[0.0, f_lo, f_hi, FS / 2], np.random.default_rng(0).uniform(0, FS / 2, 500)])
    expected = so.cosine_filterbank(n_bands, f_lo, f_hi).response(freqs)
    np.testing.assert_allclose(cosine_responses(freqs, n_bands, f_lo, f_hi), expected, atol=1e-12)


@pytest.mark.parametrize("rt60", [0.15, 0.4, 1.3])
def test_band_rt60s_and_onsets_match_sonore(rt60):
    cfs = so.cosine_filterbank(32, 20.0, 10000.0).cfs
    rt60s = band_rt60s(rt60, cfs).numpy()
    np.testing.assert_allclose(rt60s, so.band_rt60s(rt60, cfs), rtol=1e-12)
    fit_drr, _, fit_freqs = _model()
    onsets = fit_drr[:, 0] * np.log10(rt60) + fit_drr[:, 1]
    expected = np.interp(cfs, fit_freqs, onsets - np.median(onsets))
    np.testing.assert_allclose(onset_levels_db(rt60, cfs).numpy(), expected, atol=1e-12)


def _block_energy(x, bank, n_blocks, block):
    padded = np.zeros(n_blocks * block)
    padded[: min(len(x), len(padded))] = x[: len(padded)]
    bands = np.asarray(bank.analyze(so.Sound(padded, FS)).data).reshape(len(padded), -1)[:, 1:-1]
    return (bands**2).reshape(n_blocks, block, -1).sum(1).T


def test_gain_predicts_reverberant_noise_on_average():
    from scipy.signal import fftconvolve

    # A noise burst through 16 rooms: the mean of the reverberant band
    # energies matches the burst's energies convolved with the expected gain.
    # tools/room_gain_check.py measures this in full (0.2-0.3 dB mean error).
    rt60, drr, block, n_blocks = 0.4, 10.0, round(0.01 * FS), 88
    bank = so.cosine_filterbank(39, 20.0, 4000.0)
    gain = RoomGain(fs=FS, n_blocks=n_blocks, drr_db=drr).gain(torch.tensor(rt60)).numpy()
    observed, model = 0.0, 0.0
    for seed in range(16):
        burst = so.gaussian_noise(0.4, FS, rng=100 + seed).data.ravel()
        ir = so.synth_ir(rt60, FS, drr_db=drr, rng=seed).data.ravel()
        observed = observed + _block_energy(fftconvolve(burst, ir), bank, n_blocks, block)
        source = _block_energy(burst, bank, n_blocks, block)
        model = model + np.stack([np.convolve(s, g)[:n_blocks] for s, g in zip(source, gain, strict=True)])
    keep = model > model.max(1, keepdims=True) * 1e-6
    error_db = 10 * np.log10(observed[keep] / model[keep])
    assert abs(error_db.mean()) < 0.6
    assert np.abs(error_db).max() < 6.0


def test_gain_is_differentiable_and_longer_rooms_ring_longer():
    room = RoomGain(fs=FS, n_blocks=60, drr_db=10.0)
    rt60 = torch.tensor(0.4, requires_grad=True)
    late = room.gain(rt60)[:, 30:].sum()
    late.backward()
    assert torch.isfinite(rt60.grad) and rt60.grad > 0
    gain = room.gain(torch.tensor(0.4))
    assert torch.all(gain[:, 0] >= 1.0)  # the direct impulse, plus the start of the tail
