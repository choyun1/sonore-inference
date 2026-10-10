"""Marginal likelihoods of hypotheses, for enumerative inference (design D5).

Comparing hypotheses with different numbers of parameters needs the
marginal likelihood ``p(sound | H) = integral of p(sound | theta, H) p(theta | H)``,
not the best fit [App. B.1.1 of Cusimano et al., 2024]. Two estimates are
given, from the posterior mode ``theta*`` found by :func:`sonore_inference.fit.fit`
with the prior included:

- **Laplace**: a Gaussian fitted at the mode, ``log p(sound, theta*) +
  d/2 log 2 pi - 1/2 log det H``, with ``H`` the Hessian of the negative
  log joint density.
- **Importance sampling** with that Gaussian as the proposal, which is
  unbiased for ``p(sound | H)`` and checks the Laplace estimate. Its
  effective sample size says how well the proposal covers the posterior.

The paper used variational inference followed by importance sampling
[App. B.3]. :func:`variational_evidence` does the same: a Gaussian is fitted
to the posterior by maximizing the evidence lower bound (ELBO), then used as
the importance-sampling proposal. It is slower than Laplace but does not
assume the posterior is a smooth peak, which matters where the likelihood is
rugged (milestone (b): small changes to an f0 trajectory shift the relative
phase of harmonics sharing a cochleagram channel, so the curvature at the
mode changes sign from point to point).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


def flatten(
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, torch.Tensor]]]:
    """One vector holding every parameter, and the function that splits it back into a dict."""
    names = list(params)
    shapes = [params[name].shape for name in names]
    sizes = [params[name].numel() for name in names]
    vector = torch.cat([params[name].reshape(-1) for name in names])

    def unflatten(flat: torch.Tensor) -> dict[str, torch.Tensor]:
        pieces = torch.split(flat, sizes)
        return {name: piece.reshape(shape) for name, piece, shape in zip(names, pieces, shapes, strict=True)}

    return vector, unflatten


@dataclass
class Evidence:
    laplace: float
    importance: float
    effective_sample_size: float
    n_samples: int
    log_joint_at_mode: float
    floored_eigenvalues: int


def log_evidence(
    log_joint: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    mode: dict[str, torch.Tensor],
    *,
    n_samples: int = 128,
    generator: torch.Generator | None = None,
    min_eigenvalue: float = 1e-6,
) -> Evidence:
    """Laplace and importance-sampling estimates of ``log p(sound | H)``.

    ``log_joint`` maps parameters to ``log p(sound | theta) + log p(theta)``;
    ``mode`` is where it peaks. Eigenvalues of the Hessian below
    ``min_eigenvalue`` (a mode that is not quite a maximum, or a flat
    direction) are raised to it, and how many were is reported.
    """
    vector, unflatten = flatten({name: value.detach() for name, value in mode.items()})

    def flat_log_joint(flat):
        return log_joint(unflatten(flat))

    peak = flat_log_joint(vector).item()
    hessian = torch.autograd.functional.hessian(lambda flat: -flat_log_joint(flat), vector)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    floored = int((eigenvalues < min_eigenvalue).sum())
    eigenvalues = eigenvalues.clamp_min(min_eigenvalue)
    dimension = vector.numel()
    laplace = peak + 0.5 * dimension * math.log(2 * math.pi) - 0.5 * eigenvalues.log().sum().item()

    covariance = eigenvectors @ torch.diag(1 / eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    proposal = torch.distributions.MultivariateNormal(vector, covariance_matrix=covariance)
    if generator is not None:
        draws = (
            vector
            + torch.randn(n_samples, dimension, generator=generator, dtype=vector.dtype)
            @ torch.linalg.cholesky(covariance).T
        )
    else:
        draws = proposal.sample((n_samples,))

    def log_weight(draw):
        # a draw far outside the prior's support can make the model itself
        # fail (an overflowing f0, say); it carries no weight
        try:
            return flat_log_joint(draw) - proposal.log_prob(draw)
        except ValueError:
            return torch.tensor(-math.inf, dtype=vector.dtype)

    with torch.no_grad():
        log_weights = torch.stack([log_weight(draw) for draw in draws])
    finite = torch.isfinite(log_weights)
    log_weights = torch.where(finite, log_weights, torch.full_like(log_weights, -math.inf))
    importance = (torch.logsumexp(log_weights, 0) - math.log(n_samples)).item()
    normalized = torch.softmax(log_weights, 0)
    effective = (1 / (normalized**2).sum()).item()
    return Evidence(laplace, importance, effective, n_samples, peak, floored)


@dataclass
class VariationalEvidence:
    elbo: float
    importance: float
    effective_sample_size: float
    n_samples: int
    elbo_history: list[float]


def variational_evidence(
    log_joint: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    mode: dict[str, torch.Tensor],
    *,
    steps: int = 2000,
    learning_rate: float = 0.01,
    n_draws: int = 10,
    n_samples: int = 256,
    generator: torch.Generator | None = None,
    min_eigenvalue: float = 1e-2,
) -> VariationalEvidence:
    """ELBO and importance-sampling estimates of ``log p(sound | H)`` from a fitted Gaussian.

    The Gaussian ``q`` is fitted in coordinates whitened by the Laplace
    approximation at ``mode`` (absolute eigenvalues of the Hessian, raised to
    at least ``min_eigenvalue``): ``theta = mode + W z``, with ``z`` Gaussian
    with its own mean and independent scales, fitted by Adam on the ELBO with
    ``n_draws`` reparameterized draws per step. Draws outside the prior's
    support (log joint -inf) are left out of a step's average. ``elbo`` is the
    mean of the last tenth of the steps' estimates, a lower bound on the log
    evidence up to noise; ``importance`` uses ``q`` as the proposal.
    """
    vector, unflatten = flatten({name: value.detach() for name, value in mode.items()})

    def flat_log_joint(flat):
        return log_joint(unflatten(flat))

    dimension = vector.numel()
    hessian = torch.autograd.functional.hessian(lambda flat: -flat_log_joint(flat), vector)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (hessian + hessian.T))
    curvature = eigenvalues.abs().clamp_min(min_eigenvalue)
    whitening = eigenvectors * curvature.rsqrt()
    log_det_whitening = -0.5 * curvature.log().sum().item()
    mean = torch.zeros(dimension, dtype=vector.dtype, requires_grad=True)
    log_scale = torch.zeros(dimension, dtype=vector.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([mean, log_scale], lr=learning_rate)
    gaussian_entropy = 0.5 * dimension * (1 + math.log(2 * math.pi)) + log_det_whitening
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        noise = torch.randn(n_draws, dimension, generator=generator, dtype=vector.dtype)
        draws = vector + (mean + noise * log_scale.exp()) @ whitening.T
        values = []
        for draw in draws:
            with torch.no_grad():
                inside = math.isfinite(flat_log_joint(draw).item())
            if inside:
                values.append(flat_log_joint(draw))
        if not values:
            continue
        elbo = torch.stack(values).mean() + log_scale.sum() + gaussian_entropy
        (-elbo).backward()
        optimizer.step()
        history.append(elbo.item())

    with torch.no_grad():
        scale = log_scale.exp()
        noise = torch.randn(n_samples, dimension, generator=generator, dtype=vector.dtype)
        latent = mean + noise * scale
        # log q(theta) = log N(z; mean, diag(scale^2)) - log |det W|
        log_q = (
            -0.5 * (noise**2).sum(-1)
            - log_scale.sum()
            - 0.5 * dimension * math.log(2 * math.pi)
            - log_det_whitening
        )

        def log_weight(z, log_q_z):
            try:
                return flat_log_joint(vector + whitening @ z) - log_q_z
            except ValueError:
                return torch.tensor(-math.inf, dtype=vector.dtype)

        log_weights = torch.stack([log_weight(z, lq) for z, lq in zip(latent, log_q, strict=True)])
    finite = torch.isfinite(log_weights)
    log_weights = torch.where(finite, log_weights, torch.full_like(log_weights, -math.inf))
    importance = (torch.logsumexp(log_weights, 0) - math.log(n_samples)).item()
    effective = (1 / (torch.softmax(log_weights, 0) ** 2).sum()).item()
    tail = history[-max(1, len(history) // 10) :]
    return VariationalEvidence(sum(tail) / len(tail), importance, effective, n_samples, history)
