"""Laplace evidence over the source levels at fixed RT60 and source
statistics (docs/design/source-reverb.md, R5), against autograd and
:func:`sonore_inference.evidence.log_evidence`."""

import math

import numpy as np
import pytest
import sonore as so
import torch
from scipy.signal import fftconvolve

from sonore_inference.blockpower import BlockPower
from sonore_inference.evidence import log_evidence
from sonore_inference.sourceroom import (
    THETA_DEFAULT,
    Expansion,
    Model,
    Prior,
    ar1_cholesky,
    fit_mode,
    newton_mode,
)
from sonore_inference.spectrotemporal import SpectrogramPrior

FS = 20_000.0
RT60 = 0.3


@pytest.fixture(scope="module")
def model():
    """A 200 ms source in a 0.3 s room, 400 ms analysed, to keep it quick."""
    power = BlockPower(n_blocks=40, duration=0.2)
    env = so.gaussian_spectrogram(power.duration, FS, rng=0)
    x = env.to_sound(so.gaussian_noise(power.duration, FS, rng=1000)).ramp(power.ramp).data.ravel()
    ir = so.synth_ir(RT60, FS, drr_db=9.9, rng=2000).data.ravel()
    return Model(power.analyze(fftconvolve(x, ir)), power)


def log_theta(theta=THETA_DEFAULT):
    return torch.tensor(np.log(theta), dtype=torch.float64)


def test_prior_matches_the_spectrogram_prior():
    n_windows = 7
    prior, reference = Prior(log_theta(), 39, n_windows), SpectrogramPrior()
    covariance = reference.covariance(n_windows)
    identity = torch.eye(covariance.shape[0], dtype=torch.float64)
    torch.testing.assert_close(prior.precision() @ covariance, identity, atol=1e-10, rtol=0)
    assert prior.log_det_covariance().item() == pytest.approx(torch.logdet(covariance).item(), abs=1e-8)
    z = torch.randn(39, n_windows, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    torch.testing.assert_close(prior.color(z), reference.color(z), atol=1e-10, rtol=0)


def test_likelihood_derivatives_are_exact(model):
    gain = model.gain(RT60)
    levels = model.mean + 10 * torch.randn(*model.shape, generator=torch.Generator().manual_seed(1))

    def log_likelihood(flat):
        return model.log_likelihood(flat.reshape(model.shape), gain)

    value, gradient, hessian = model.likelihood_derivatives(levels, gain)
    flat = levels.reshape(-1)
    assert value.item() == pytest.approx(log_likelihood(flat).item(), rel=1e-12)
    torch.testing.assert_close(gradient, torch.func.grad(log_likelihood)(flat), rtol=1e-8, atol=1e-8)
    reference = -torch.autograd.functional.hessian(log_likelihood, flat)
    torch.testing.assert_close(hessian, reference, rtol=1e-8, atol=1e-9)


def test_evidence_gradient_in_theta_is_exact():
    n = 39 * 5
    generator = torch.Generator().manual_seed(2)
    a = 0.05 * torch.randn(n, n, generator=generator, dtype=torch.float64)
    expansion = Expansion(
        levels=5 * torch.randn(39, 5, generator=generator, dtype=torch.float64),
        value=-100.0,
        gradient=torch.randn(n, generator=generator, dtype=torch.float64),
        hessian=a @ a.T,
        mean=torch.zeros(n, dtype=torch.float64),
    )
    x = np.log([6.0, 0.1, 10.0])
    value, gradient = expansion.value_and_gradient(x)
    t = torch.tensor(x, requires_grad=True)
    reference = expansion.log_evidence(t)
    reference.backward()
    assert value == pytest.approx(reference.item(), rel=1e-12)
    np.testing.assert_allclose(gradient, t.grad.numpy(), rtol=1e-8)


def test_laplace_at_the_mode_equals_log_evidence(model):
    # The expansion at the mode is the Laplace approximation that
    # evidence.log_evidence computes with autograd, here in whitened coordinates.
    gain = model.gain(RT60)
    prior = Prior(log_theta(), *model.shape)
    levels, expansion = newton_mode(model, gain, prior, fit_mode(model, gain, prior))
    lb = ar1_cholesky(prior.rho_band, model.shape[0])
    lt = ar1_cholesky(prior.rho_time, model.shape[1])
    z = torch.linalg.solve_triangular(lb, (levels - model.mean) / prior.sd, upper=False)
    z = torch.linalg.solve_triangular(lt, z.T, upper=False).T

    def log_joint(params):
        z = params["z"]
        deviation = prior.sd * lb @ z @ lt.T
        normal = -0.5 * (z**2).sum() - 0.5 * z.numel() * math.log(2 * math.pi)
        return model.log_likelihood(model.mean + deviation, gain) + normal

    reference = log_evidence(log_joint, {"z": z}, n_samples=4, generator=torch.Generator().manual_seed(3))
    assert reference.floored_eigenvalues == 0
    assert expansion.log_evidence(log_theta()).item() == pytest.approx(reference.laplace, abs=1e-2)
