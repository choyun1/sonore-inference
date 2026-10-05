"""Point estimates by gradient ascent on the likelihood (design D5, first step).

A hypothesis is a function from named parameter tensors to a waveform. The
fit maximizes the Gaussian log likelihood of the observed cochleagram with
Adam, from a given initialization. There are no priors yet, so this is a
maximum-likelihood fit, and the best log likelihoods of hypotheses with
different numbers of parameters are not comparable as evidence: the larger
hypothesis can only do as well or better (App. B.1.1 of Cusimano et al.,
2024). Turning them into a comparison needs priors and a marginal
likelihood, which come with enumerative inference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood


@dataclass
class FitResult:
    params: dict[str, torch.Tensor]
    log_likelihood: float
    history: list[float] = field(default_factory=list)


def fit(
    render: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    init: dict[str, torch.Tensor],
    observed: torch.Tensor,
    cochleagram: Cochleagram,
    *,
    learning_rates: dict[str, float],
    steps: int = 300,
    sigma: float = 10.0,
) -> FitResult:
    """Maximize ``log p(observed | render(params))`` over ``params`` with Adam.

    ``init`` gives each parameter's starting value and ``learning_rates``
    its step size (Adam's steps are roughly that size in the parameter's own
    units, so dB and log-frequency parameters want different ones). Returns
    the parameters with the best log likelihood seen, which is evaluated
    before every step.
    """
    params = {name: value.detach().clone().requires_grad_() for name, value in init.items()}
    optimizer = torch.optim.Adam([{"params": [params[name]], "lr": learning_rates[name]} for name in params])
    best = FitResult({name: value.detach().clone() for name, value in params.items()}, float("-inf"))
    for _ in range(steps + 1):
        optimizer.zero_grad()
        log_likelihood = gaussian_log_likelihood(observed, cochleagram(render(params)), sigma)
        value = log_likelihood.item()
        best.history.append(value)
        if value > best.log_likelihood:
            best.log_likelihood = value
            best.params = {name: tensor.detach().clone() for name, tensor in params.items()}
        (-log_likelihood).backward()
        optimizer.step()
    return best
