"""Priors against closed forms, and evidence estimates on a case with a known answer."""

import math

import numpy as np
import pytest
import torch

from sonore_inference.cochleagram import erb_to_freq, freq_to_erb
from sonore_inference.evidence import log_evidence
from sonore_inference.priors import (
    log_prior_level,
    log_prior_log_frequency,
    log_prior_spectrum,
    log_prior_structure,
)


def test_frequency_prior_is_uniform_in_erb_number():
    # integrate exp(log density) over log f across the ERB range: it must be 1
    log_freqs = torch.linspace(
        math.log(erb_to_freq(3.0)), math.log(erb_to_freq(33.0)), 200_001, dtype=torch.float64
    )
    density = log_prior_log_frequency(log_freqs).exp()
    assert torch.trapezoid(density, log_freqs).item() == pytest.approx(1.0, abs=1e-6)
    # equal ERB intervals carry equal mass
    for low, high in [(5.0, 6.0), (25.0, 26.0)]:
        grid = torch.linspace(
            math.log(erb_to_freq(low)), math.log(erb_to_freq(high)), 20_001, dtype=torch.float64
        )
        assert torch.trapezoid(log_prior_log_frequency(grid).exp(), grid).item() == pytest.approx(
            1 / 30, rel=1e-6
        )
    assert log_prior_log_frequency(torch.tensor(math.log(erb_to_freq(34.0)))).item() == -math.inf


def test_level_prior():
    assert log_prior_level(torch.tensor(60.0)).item() == pytest.approx(-math.log(120))
    assert log_prior_level(torch.tensor(-1.0)).item() == -math.inf


def test_spectrum_prior_matches_numpy():
    f0 = 200.0
    spectrum = torch.tensor([3.0, -2.0, 0.5, 1.0], dtype=torch.float64)
    erbs = freq_to_erb(f0 * np.arange(1, 5))
    covariance = 11.8**2 * np.exp(-0.5 * ((erbs[:, None] - erbs[None, :]) / 4.7) ** 2) + 0.5**2 * np.eye(4)
    x = spectrum.numpy()
    expected = -0.5 * (x @ np.linalg.solve(covariance, x) + np.linalg.slogdet(2 * np.pi * covariance)[1])
    assert log_prior_spectrum(spectrum, torch.tensor(f0, dtype=torch.float64)).item() == pytest.approx(
        expected
    )


def test_structure_prior():
    rate = 0.5  # 1 source per second over 0.5 s
    truncation = -math.log1p(-math.exp(-rate))
    one = math.log(rate) - rate + truncation - math.log(3) + math.log(0.5)
    two = (
        2 * math.log(rate)
        - rate
        - math.log(2)
        + truncation
        + math.log(2)
        - 2 * math.log(3)
        + 2 * math.log(0.5)
    )
    assert log_prior_structure(["harmonic"], 0.5) == pytest.approx(one)
    assert log_prior_structure(["harmonic", "whistle"], 0.5) == pytest.approx(two)
    # two sources of the same type can be drawn in only one order
    assert log_prior_structure(["whistle", "whistle"], 0.5) == pytest.approx(two - math.log(2))


def test_evidence_of_a_gaussian_model():
    # y_i ~ N(theta, 1), theta ~ N(0, 4): the marginal likelihood is a closed form, and Laplace is exact
    y = torch.tensor([0.3, 1.1, 0.7, 1.9], dtype=torch.float64)
    zero, one, two = (torch.tensor(v, dtype=torch.float64) for v in (0.0, 1.0, 2.0))

    def log_joint(params):
        theta = params["theta"]
        likelihood = torch.distributions.Normal(theta, one).log_prob(y).sum()
        return likelihood + torch.distributions.Normal(zero, two).log_prob(theta).sum()

    n = len(y)
    posterior_mean = y.sum() / (n + 1 / 4)
    covariance = torch.eye(n, dtype=torch.float64) + 4.0
    exact = (
        torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), covariance)
        .log_prob(y)
        .item()
    )
    result = log_evidence(
        log_joint,
        {"theta": posterior_mean.reshape(1)},
        n_samples=64,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.laplace == pytest.approx(exact, abs=1e-12)
    # the proposal is the posterior, so every importance weight is equal
    assert result.importance == pytest.approx(exact, abs=1e-12)
    assert result.effective_sample_size == pytest.approx(64)
    assert result.floored_eigenvalues == 0
