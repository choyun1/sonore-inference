"""Block band energies and their expected values for a random spectrogram
(docs/design/source-reverb.md, R3-R4), against sonore."""

import numpy as np
import pytest
import sonore as so
import torch
from scipy.special import hyp2f1
from sonore.core.utils import as_rng
from sonore.sources.gaussian_spectrogram import _correlated_field

from sonore_inference.blockpower import BlockPower, log_likelihood, phase_correlation
from sonore_inference.spectrotemporal import SpectrogramPrior

FS = 20_000.0


@pytest.fixture(scope="module")
def short():
    """A 200 ms source in 400 ms, to keep the precomputation quick."""
    return BlockPower(n_blocks=40, duration=0.2)


def drawn_levels(power, seed):
    prior = SpectrogramPrior()
    field = _correlated_field(prior.n_bands, power.n_windows, prior.rho_band, prior.rho_time, as_rng(seed))
    return prior.mean_db()[:, None] + prior.sd_db * field


def test_phase_correlation_is_the_hypergeometric_function():
    z = np.concatenate([np.linspace(0, 1, 501), 1 - np.logspace(-15, -1, 30)])
    np.testing.assert_allclose(phase_correlation(z), hyp2f1(0.5, 0.5, 2, z), rtol=1e-11)


def test_analyze_matches_sonore(short):
    x = so.gaussian_noise(0.3, FS, rng=3).data.ravel()
    padded = np.zeros(short.n_samples)
    padded[: len(x)] = x
    bands = np.asarray(so.cosine_filterbank(39, 20, 4000).analyze(so.Sound(padded, FS), pad=0.5).data)
    bands = bands.reshape(short.n_samples, -1)[:, 1:-1]
    expected = (bands**2).reshape(short.n_blocks, short.block_samples, -1).sum(1).T
    np.testing.assert_allclose(short.analyze(x).numpy(), expected, rtol=1e-9)


def test_n_windows_matches_sonore(short):
    env = so.gaussian_spectrogram(short.duration, FS, rng=0)
    levels = drawn_levels(short, 0)
    centers = np.asarray(env.data).reshape(env.data.shape[0], -1)[:: short.block_samples, 1:-1].T
    np.testing.assert_allclose(20 * np.log10(centers), levels[:, : centers.shape[1]], atol=1e-9)
    assert short.n_windows == centers.shape[1] + 1  # the last window is centred on the end


def test_fast_source_energy_equals_the_direct_one(short):
    levels = torch.from_numpy(drawn_levels(short, 4))
    fast, direct = short.source_energy(levels), short.source_energy_direct(levels)
    relevant = direct > direct.max() * 1e-9
    torch.testing.assert_close(fast[relevant], direct[relevant], rtol=1e-8, atol=0)


def test_source_energy_matches_renders_on_average(short):
    # 16 carriers on one spectrogram: the mean of the renders' block energies
    # against the model, over cells within 60 dB of the peak. Measured in full
    # by tools/block_power_check.py.
    seed = 2
    env = so.gaussian_spectrogram(short.duration, FS, rng=seed)
    model = short.source_energy(torch.from_numpy(drawn_levels(short, seed))).numpy()
    carriers = [so.gaussian_noise(short.duration, FS, rng=100 + s) for s in range(16)]
    renders = [short.analyze(env.to_sound(c).ramp(short.ramp).data.ravel()) for c in carriers]
    observed = torch.stack(renders).mean(0).numpy()
    keep = model > model.max() * 1e-6
    error_db = 10 * np.log10(observed[keep] / model[keep])
    playing = np.broadcast_to(np.arange(short.n_blocks) < 20, model.shape)[keep]
    assert abs(error_db[playing].mean()) < 0.5
    assert abs(error_db[~playing].mean()) < 1.5  # the filters' ringing after the offset


def test_log_likelihood_is_gaussian_in_db_with_floor():
    observed = torch.tensor([[1.0, 1e-3, 1e-9]])
    expected = torch.tensor([[10**0.1, 1e-3, 1e-12]])
    sigma = 2.0
    # residuals in dB: -1, 0, and both below the 40 dB floor, so 0
    value = log_likelihood(observed, expected, sigma, floor_db=40.0, bias_db=0.0)
    gauss = -0.5 * (np.array([1.0, 0.0, 0.0]) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    assert value.item() == pytest.approx(gauss.sum())
    shifted = log_likelihood(observed, expected / 10**0.1, sigma, floor_db=40.0, bias_db=1.0)
    assert shifted.item() == pytest.approx(value.item())


def test_source_energy_is_differentiable(short):
    levels = torch.from_numpy(drawn_levels(short, 1)).requires_grad_(True)
    short.source_energy(levels)[:, 10].sum().backward()
    assert torch.isfinite(levels.grad).all()
    near, far = levels.grad[:, 10], levels.grad[:, 20]
    assert (near > 0).all()
    assert (far.abs() < 1e-3 * near.abs().max()).all()  # only the filters' ringing reaches 100 ms
