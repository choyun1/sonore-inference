"""The spectrogram prior against sonore's gaussian_spectrogram and against a
dense Gaussian (docs/design/source-reverb.md, R1)."""

import numpy as np
import pytest
import sonore as so
import torch

from sonore_inference.spectrotemporal import SpectrogramPrior

FS = 20000.0


def drawn_levels_db(duration, seed, prior):
    """The cell levels sonore drew, read off its envelopes at the window
    centers, where only that window's raised cosine is nonzero."""
    env = so.gaussian_spectrogram(duration, FS, rng=seed)
    data = np.asarray(env.data).reshape(env.data.shape[0], prior.n_bands + 2)
    hop = round(prior.step * FS)
    centers = np.arange(0, data.shape[0], hop)
    return 20 * np.log10(data[centers, 1:-1].T)


def test_spacing_matches_sonore():
    prior = SpectrogramPrior()
    assert prior.band_spacing_erb == pytest.approx(so.cosine_filterbank(39, 20.0, 4000.0).spacing, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_whitening_recovers_sonores_normals(seed):
    # Same mean, spacing, correlations and recursion as sonore: whitening its
    # drawn levels gives back the standard normals it started from. Its last
    # window center lies past the end of the sound, so the grid is a prefix.
    prior = SpectrogramPrior()
    levels = drawn_levels_db(0.4, seed, prior)
    deviation = torch.from_numpy(levels - prior.mean_db()[:, None])
    normals = np.random.default_rng(seed).standard_normal((prior.n_bands, 41))
    np.testing.assert_allclose(prior.whiten(deviation).numpy(), normals[:, : levels.shape[1]], atol=1e-9)


def test_color_inverts_whiten():
    prior = SpectrogramPrior()
    z = torch.randn(prior.n_bands, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(prior.whiten(prior.color(z)), z)


def test_log_prob_matches_dense_gaussian():
    prior = SpectrogramPrior(n_bands=5, band_correlation_erb=3.0, time_correlation=0.05, sd_db=6.0)
    n_windows = 7
    dense = torch.distributions.MultivariateNormal(
        torch.zeros(5 * n_windows, dtype=torch.float64), prior.covariance(n_windows)
    )
    x = dense.sample(torch.Size([3]))
    for row in x:
        torch.testing.assert_close(prior.log_prob(row.reshape(5, n_windows)), dense.log_prob(row))
