"""Priors for the constant-parameter harmonic and whistle sources.

They follow the paper's source priors [App. A.3-A.4 and Tables A.1-A.2 of
Cusimano et al., 2024], simplified to match our sources, whose frequency and
level do not change over an event:

- **f0** (harmonic) and **frequency** (whistle): uniform in ERB number on
  [3, 33] Cams, the bounds of the paper's uniform prior on the mean of the
  f0 Gaussian process. With f0 parameterized as ``log f0``, the density
  carries the Jacobian ``d(ERB number)/d(log f0)``, about ``f0 / ERB(f0)``.
- **Overall level**: uniform on [0, 120] dB, the paper's bounds for the
  amplitude mean of whistles and harmonic sources.
- **Harmonic spectrum**: a zero-mean Gaussian process over the ERB numbers of
  the harmonics, squared-exponential kernel, with the paper's median fitted
  hyperparameters (sigma 11.8 dB, lengthscale 4.7 ERB) and its jitter
  (epsilon 0.5). The paper puts a prior on sigma and the lengthscale; here
  they are fixed at the medians, a simplification.
- **Number of sources**: Poisson with 1 source per second times the
  scene's duration, zero-truncated (design D8, option A); each source's type
  uniform over noise, harmonic and whistle; each source's number of events
  Geometric(0.5) [Table A.1].

In the functions above, the paper's Gaussian processes over time collapse
to their means. :class:`TrajectoryPrior` restores them, for sources whose
frequency and level follow trajectories; :func:`log_prior_event_timing` is
the prior of an event's onset and duration, for when they are inferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ERB_RANGE = (3.0, 33.0)
LEVEL_RANGE_DB = (0.0, 120.0)
SPECTRUM_SIGMA_DB = 11.8
SPECTRUM_LENGTHSCALE_ERB = 4.7
SPECTRUM_JITTER_DB = 0.5
SOURCES_PER_SECOND = 1.0
EVENTS_GEOMETRIC_P = 0.5
N_SOURCE_TYPES = 3


def _freq_to_erb(freq: torch.Tensor) -> torch.Tensor:
    return 9.265 * torch.log1p(freq / (24.7 * 9.265))


def log_prior_log_frequency(log_freq: torch.Tensor) -> torch.Tensor:
    """Log density of ``log f`` when f is uniform in ERB number on ``ERB_RANGE``; -inf outside."""
    freq = log_freq.exp()
    erb = _freq_to_erb(freq)
    low, high = ERB_RANGE
    inside = (erb >= low) & (erb <= high)
    # d(ERB number)/d(log f) = f * 9.265 / (24.7 * 9.265 + f), the exact derivative of _freq_to_erb
    density = -math.log(high - low) + log_freq + math.log(9.265) - torch.log(24.7 * 9.265 + freq)
    return torch.where(inside, density, torch.full_like(density, -math.inf))


def log_prior_level(level_db: torch.Tensor) -> torch.Tensor:
    """Log density of a level uniform on ``LEVEL_RANGE_DB``; -inf outside."""
    low, high = LEVEL_RANGE_DB
    inside = (level_db >= low) & (level_db <= high)
    density = torch.full_like(level_db, -math.log(high - low))
    return torch.where(inside, density, torch.full_like(level_db, -math.inf))


def spectrum_distribution(n_harmonics: int, f0: torch.Tensor) -> torch.distributions.MultivariateNormal:
    """The GP prior of a harmonic spectrum (dB, one value per harmonic 1..K),
    with the kernel evaluated at the ERB numbers of the harmonics of ``f0``."""
    numbers = torch.arange(1, n_harmonics + 1, dtype=f0.dtype, device=f0.device)
    erbs = _freq_to_erb(numbers * f0)
    distance = erbs[:, None] - erbs[None, :]
    covariance = SPECTRUM_SIGMA_DB**2 * torch.exp(-0.5 * (distance / SPECTRUM_LENGTHSCALE_ERB) ** 2)
    covariance = covariance + SPECTRUM_JITTER_DB**2 * torch.eye(n_harmonics, dtype=f0.dtype, device=f0.device)
    mean = torch.zeros(n_harmonics, dtype=f0.dtype, device=f0.device)
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)


def log_prior_spectrum(spectrum_db: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Log density of a harmonic spectrum under :func:`spectrum_distribution`."""
    f0 = torch.as_tensor(f0, dtype=spectrum_db.dtype, device=spectrum_db.device)
    return spectrum_distribution(spectrum_db.shape[-1], f0).log_prob(spectrum_db)


def log_prior_structure(source_types: list[str], duration: float, n_events: list[int] | None = None) -> float:
    """Log prior of a scene's discrete structure: how many sources, their types and events.

    ``source_types`` is the unordered list of types, e.g. ``["harmonic", "whistle"]``;
    ``n_events`` the number of events of each (one each by default), each
    Geometric(0.5) on 1, 2, ... [Table A.1].
    The number of sources is zero-truncated Poisson(rate * duration). A fitted
    hypothesis is one mode with labelled sources; every permutation of the
    labels is another mode of equal mass that the same hypothesis includes,
    so the prior counts all n! labellings (design C6). For sources of the same
    type this is more than the number of type orders.
    """
    n = len(source_types)
    rate = SOURCES_PER_SECOND * duration
    log_poisson = n * math.log(rate) - rate - math.lgamma(n + 1)
    log_truncation = -math.log1p(-math.exp(-rate))
    log_labellings = math.lgamma(n + 1)
    log_types = -n * math.log(N_SOURCE_TYPES)
    extra_events = 0 if n_events is None else sum(n_events) - n
    log_events = n * math.log(EVENTS_GEOMETRIC_P) + extra_events * math.log1p(-EVENTS_GEOMETRIC_P)
    return log_poisson + log_truncation + log_labellings + log_types + log_events


class ZeroTruncatedPoisson(torch.distributions.Distribution):
    """The number of sources: Poisson(``rate``) conditioned on at least one (design D8)."""

    arg_constraints = {}
    support = torch.distributions.constraints.positive_integer

    def __init__(self, rate: float):
        self.rate = rate
        super().__init__(validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return (
            value * math.log(self.rate)
            - self.rate
            - torch.lgamma(value + 1)
            - math.log1p(-math.exp(-self.rate))
        )

    def sample(self, sample_shape=()) -> torch.Tensor:
        poisson = torch.distributions.Poisson(torch.tensor(float(self.rate)))
        value = poisson.sample(torch.Size(sample_shape))
        while (value == 0).any():
            value = torch.where(value == 0, poisson.sample(torch.Size(sample_shape)), value)
        return value


SOURCE_TYPES = ("noise", "harmonic", "whistle")


def sample_structure(duration: float, exclude: tuple[str, ...] = ()) -> list[tuple[str, int]]:
    """Draw (type, number of events) for each source from the prior that
    :func:`log_prior_structure` scores, using torch's global random state.

    Types in ``exclude`` are rejected by drawing the whole structure again, so
    the result is the prior conditioned on not containing them.
    """
    n_sources = ZeroTruncatedPoisson(SOURCES_PER_SECOND * duration)
    types = torch.distributions.Categorical(torch.ones(N_SOURCE_TYPES))
    # torch's Geometric counts failures before the first success: m - 1
    events = torch.distributions.Geometric(torch.tensor(EVENTS_GEOMETRIC_P))
    while True:
        n = int(n_sources.sample())
        structure = [(SOURCE_TYPES[int(types.sample())], int(events.sample()) + 1) for _ in range(n)]
        if not any(kind in exclude for kind, _ in structure):
            return structure


TIMING_NORMAL_GAMMA = dict(mu=-1.0, lam=0.5, alpha=2.5, beta=1.0)


class BoundedUniform(torch.distributions.Uniform):
    """Uniform on [low, high], ends included: log density -log(high - low) inside, -inf outside.

    The same numbers as the priors here have always computed, as a
    distribution object, so that one definition serves scoring, sampling and
    Pyro (design C7).
    """

    def __init__(self, low: float, high: float, dtype=torch.float64):
        super().__init__(torch.tensor(low, dtype=dtype), torch.tensor(high, dtype=dtype), validate_args=False)
        self.bounds = (low, high)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        low, high = self.bounds
        inside = (value >= low) & (value <= high)
        return torch.where(
            inside, torch.full_like(value, -math.log(high - low)), torch.full_like(value, -math.inf)
        )


def log_duration_distribution(dtype=torch.float64) -> torch.distributions.StudentT:
    """Prior of one event's log duration [log s], or of one rest: the log-normal with
    the normal-gamma of Table A.1 integrated out, a Student-t (see
    :func:`log_prior_event_timing`)."""
    p = TIMING_NORMAL_GAMMA
    scale = math.sqrt(p["beta"] * (p["lam"] + 1) / (p["alpha"] * p["lam"]))
    return torch.distributions.StudentT(
        torch.tensor(2 * p["alpha"], dtype=dtype),
        torch.tensor(p["mu"], dtype=dtype),
        torch.tensor(scale, dtype=dtype),
    )


class NormalGammaMarginal(torch.distributions.Distribution):
    """``k`` values x_i ~ Normal(mu, 1 / tau) that share (mu, tau) ~ NormalGamma(mu0, lam0, alpha0, beta0),
    with (mu, tau) integrated out in closed form (design C3 (D)).

    This is the prior of one source's log durations, or of its log rests
    [App. A.2, Eqns A.9-A.10]: they share the source's (mu, lambda), drawn
    from the normal-gamma of Table A.1. For ``k = 1`` it is the Student-t of
    :func:`log_duration_distribution`. The density is the conjugate
    marginal likelihood: with mean m and sum of squares S of the x_i,
    lam_k = lam0 + k, alpha_k = alpha0 + k / 2 and
    beta_k = beta0 + S / 2 + lam0 k (m - mu0)^2 / (2 lam_k),
    log p = lgamma(alpha_k) - lgamma(alpha0) + alpha0 log beta0 - alpha_k log beta_k
    + log(lam0 / lam_k) / 2 - k log(2 pi) / 2.
    """

    arg_constraints = {}
    support = torch.distributions.constraints.real_vector

    def __init__(self, k: int, dtype=torch.float64):
        self.k = k
        self.dtype = dtype
        super().__init__(event_shape=torch.Size([k]), validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        p, k = TIMING_NORMAL_GAMMA, self.k
        mean = value.mean(-1)
        sum_of_squares = ((value - mean[..., None]) ** 2).sum(-1)
        lam_k = p["lam"] + k
        alpha_k = p["alpha"] + k / 2
        beta_k = p["beta"] + sum_of_squares / 2 + p["lam"] * k * (mean - p["mu"]) ** 2 / (2 * lam_k)
        return (
            math.lgamma(alpha_k)
            - math.lgamma(p["alpha"])
            + p["alpha"] * math.log(p["beta"])
            - alpha_k * torch.log(beta_k)
            + 0.5 * math.log(p["lam"] / lam_k)
            - 0.5 * k * math.log(2 * math.pi)
        )

    def sample(self, sample_shape=()) -> torch.Tensor:
        p = TIMING_NORMAL_GAMMA
        shape = torch.Size(sample_shape)
        tau = torch.distributions.Gamma(
            torch.tensor(p["alpha"], dtype=self.dtype), torch.tensor(p["beta"], dtype=self.dtype)
        ).sample(shape)
        mu = p["mu"] + torch.randn(shape, dtype=self.dtype) / torch.sqrt(p["lam"] * tau)
        noise = torch.randn(shape + (self.k,), dtype=self.dtype)
        return mu[..., None] + noise / torch.sqrt(tau)[..., None]


def log_prior_event_timing(
    onset: torch.Tensor, log_duration: torch.Tensor, scene_duration: float
) -> torch.Tensor:
    """Log prior of a source's first (here only) event: its onset and its log duration [s].

    The onset is uniform over the scene [App. A.2]. The duration is
    log-normal with mean and precision drawn from the normal-gamma source
    prior of Table A.1 (mu0 -1, lambda0 0.5, alpha0 2.5, beta0 1); integrating
    those out leaves a Student-t on the log duration with 2 alpha0 degrees of
    freedom, location mu0 and squared scale beta0 (lambda0 + 1) / (alpha0 lambda0).
    """
    p = TIMING_NORMAL_GAMMA
    scale = math.sqrt(p["beta"] * (p["lam"] + 1) / (p["alpha"] * p["lam"]))
    student = torch.distributions.StudentT(
        torch.tensor(2 * p["alpha"], dtype=log_duration.dtype),
        torch.tensor(p["mu"], dtype=log_duration.dtype),
        torch.tensor(scale, dtype=log_duration.dtype),
    )
    inside = (onset >= 0) & (onset <= scene_duration)
    log_onset = torch.where(
        inside, torch.full_like(onset, -math.log(scene_duration)), torch.full_like(onset, -math.inf)
    )
    return log_onset + student.log_prob(log_duration)


GRID_STEP = 0.01  # s, BASS's trajectory grid (config steps.t)


@dataclass(frozen=True)
class TrajectoryPrior:
    """Gaussian-process prior of one excitation trajectory over time [App. A.2, Eqns A.13-A.17].

    The trajectory is its mean (a latent, uniform on ``mean_range``) plus a
    zero-mean Gaussian process on the 10 ms grid, with BASS's kernel: squared
    exponential (``sigma``, ``lengthscale`` [s]) plus ``beta`` squared within
    the event, plus ``epsilon`` squared and ``sigma`` times ``stability`` on
    the diagonal. Here there is one event, and the grid covers the whole
    scene; grid points outside the event are not heard, so they integrate out
    and leave the prior of the points inside it. ``sigma`` and ``lengthscale``
    are fixed at the medians of Table A.2, a simplification (the paper infers
    them). The latents are the mean and the deviations from it at each grid
    point, in the trajectory's own units, so a fit's step sizes mean the same
    along fast and slow changes.
    """

    sigma: float
    lengthscale: float
    beta: float
    epsilon: float
    mean_range: tuple[float, float]
    stability: float = 0.001

    def cholesky(self, n_grid: int, dtype=torch.float64) -> torch.Tensor:
        t = torch.arange(n_grid, dtype=dtype) * GRID_STEP
        covariance = self.sigma**2 * torch.exp(-0.5 * ((t[:, None] - t[None, :]) / self.lengthscale) ** 2)
        covariance = covariance + self.beta**2
        covariance = covariance + (self.epsilon**2 + self.sigma * self.stability) * torch.eye(
            n_grid, dtype=dtype
        )
        return torch.linalg.cholesky(covariance)

    def soft_cholesky(self, memberships: torch.Tensor) -> torch.Tensor:
        """Cholesky factor when the source has several events.

        ``memberships`` has shape ``(n_events, n_grid)``: how much each grid
        point lies inside each event (the event's gate, between 0 and 1). The
        within-event term of the kernel [App. A.2, Eqn A.17] is then
        ``beta^2 sum_j w_j(t1) w_j(t2)`` instead of a hard 0 or 1, so the
        prior moves smoothly with the events' timing (design C4, a labelled
        departure from the paper and from BASS's hard mask).
        """
        n_grid = memberships.shape[-1]
        dtype = memberships.dtype
        t = torch.arange(n_grid, dtype=dtype) * GRID_STEP
        covariance = self.sigma**2 * torch.exp(-0.5 * ((t[:, None] - t[None, :]) / self.lengthscale) ** 2)
        covariance = covariance + self.beta**2 * memberships.T @ memberships
        diagonal = self.epsilon**2 + self.sigma * self.stability
        covariance = covariance + diagonal * torch.eye(n_grid, dtype=dtype)
        return torch.linalg.cholesky(covariance)

    def mean_distribution(self, dtype=torch.float64) -> BoundedUniform:
        return BoundedUniform(*self.mean_range, dtype=dtype)

    def deviation_distribution(
        self, n_grid: int, memberships: torch.Tensor | None = None, dtype=torch.float64
    ) -> torch.distributions.MultivariateNormal:
        """The Gaussian process of the deviations on the grid. With one event
        (``memberships`` None) the within-event term covers the whole grid,
        as before: the grid points outside the event are not heard, so they
        integrate out."""
        scale_tril = self.cholesky(n_grid, dtype) if memberships is None else self.soft_cholesky(memberships)
        return torch.distributions.MultivariateNormal(torch.zeros(n_grid, dtype=dtype), scale_tril=scale_tril)

    def trajectory(self, mean: torch.Tensor, deviation: torch.Tensor) -> torch.Tensor:
        return mean + deviation

    def log_prior(self, mean: torch.Tensor, deviation: torch.Tensor) -> torch.Tensor:
        """Log density of the mean (uniform) and the deviations (the Gaussian process)."""
        low, high = self.mean_range
        inside = (mean >= low) & (mean <= high)
        log_mean = torch.where(
            inside, torch.full_like(mean, -math.log(high - low)), torch.full_like(mean, -math.inf)
        )
        scale_tril = self.cholesky(deviation.shape[-1], deviation.dtype)
        gaussian = torch.distributions.MultivariateNormal(torch.zeros_like(deviation), scale_tril=scale_tril)
        return log_mean + gaussian.log_prob(deviation)


# Table A.2 medians (Q2) of sigma and lengthscale; beta and epsilon as in
# BASS's config (full_enumerative.yaml); the f0 prior (in ERB number) is
# shared by whistles and harmonic sources.
F0_TRAJECTORY = TrajectoryPrior(sigma=5.9, lengthscale=2.5, beta=0.437, epsilon=0.1, mean_range=(3.0, 33.19))
WHISTLE_LEVEL_TRAJECTORY = TrajectoryPrior(
    sigma=1.3, lengthscale=6.9, beta=0.542, epsilon=0.1, mean_range=LEVEL_RANGE_DB
)
HARMONIC_LEVEL_TRAJECTORY = TrajectoryPrior(
    sigma=7.8, lengthscale=0.18, beta=3.12, epsilon=0.5, mean_range=LEVEL_RANGE_DB
)
