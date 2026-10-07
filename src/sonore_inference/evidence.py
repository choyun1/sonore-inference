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
[App. B.3]; the Laplace proposal here replaces the variational one.
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
