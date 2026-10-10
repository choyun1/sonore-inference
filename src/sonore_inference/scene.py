"""Scenes: hypotheses built from any number of harmonic sources and whistles.

A hypothesis in enumerative inference [App. C.6 of Cusimano et al., 2024]
is a list of sources, each with one event. :class:`Scene` holds such a list
and gives what :func:`sonore_inference.fit.fit` and
:func:`sonore_inference.evidence.log_evidence` need: a renderer from named
parameters to a waveform, the log prior of those parameters, their step
sizes, and the log prior of the scene's discrete structure.

The sources are BASS's full ones, as milestone (a) ended up using them
(``tools/compare_mistuned_harmonic.py --trajectories --harmonic-timing
inferred --whistle-timing inferred``):

- :class:`Harmonic`: f0 and overall level follow Gaussian-process
  trajectories on a 10 ms grid, plus a spectrum (dB per harmonic) whose
  prior is evaluated at the mean of the f0 trajectory (in ERB number); the number of
  harmonics is given.
- :class:`Whistle`: frequency and level follow trajectories.

Both infer their onset and log duration under
:func:`~sonore_inference.priors.log_prior_event_timing`. Parameter names
are ``"<source name>.<parameter>"``, e.g. ``"h0.f0_mean"``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood
from sonore_inference.evidence import Evidence, VariationalEvidence, log_evidence, variational_evidence
from sonore_inference.fit import FitResult, fit
from sonore_inference.priors import (
    F0_TRAJECTORY,
    GRID_STEP,
    HARMONIC_LEVEL_TRAJECTORY,
    WHISTLE_LEVEL_TRAJECTORY,
    log_prior_event_timing,
    log_prior_spectrum,
    log_prior_structure,
)
from sonore_inference.sources import trajectory_event

Params = dict[str, torch.Tensor]


def erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    return 24.7 * 9.265 * torch.expm1(erb / 9.265)


def hz_to_erb(freq: float) -> float:
    return 9.265 * math.log1p(freq / (24.7 * 9.265))


@dataclass(frozen=True)
class SceneTiming:
    """The sound's sampling rate [Hz], total duration [s] and the events' ramp [s]."""

    fs: float
    total_duration: float
    ramp: float

    @property
    def n_grid(self) -> int:
        return round(self.total_duration / GRID_STEP) + 1


def _event(timing: SceneTiming, freqs, levels, onset, duration) -> torch.Tensor:
    return trajectory_event(
        freqs,
        levels,
        onset,
        duration,
        grid_step=GRID_STEP,
        fs=timing.fs,
        total_duration=timing.total_duration,
        ramp=timing.ramp,
    )


def _timing_init(onset: float, duration: float, dtype) -> dict[str, torch.Tensor]:
    return {
        "onset": torch.tensor(onset, dtype=dtype),
        "log_duration": torch.tensor(math.log(duration), dtype=dtype),
    }


@dataclass(frozen=True)
class Harmonic:
    """A harmonic source with ``n_harmonics`` harmonics of a time-varying f0."""

    name: str
    n_harmonics: int
    kind: ClassVar[str] = "harmonic"
    # step sizes, in ERB number, dB, seconds and log seconds (those of milestone (a))
    RATES: ClassVar[dict[str, float]] = {
        "f0_mean": 5e-3,
        "f0_deviation": 2e-3,
        "level_mean": 0.5,
        "level_deviation": 0.1,
        "spectrum_db": 0.5,
        "onset": 2e-4,
        "log_duration": 1e-3,
    }

    def key(self, parameter: str) -> str:
        return f"{self.name}.{parameter}"

    def initial(
        self,
        timing: SceneTiming,
        *,
        f0: float,
        level_db: float,
        onset: float,
        duration: float,
        spectrum_db: torch.Tensor | None = None,
        dtype=torch.float64,
    ) -> Params:
        """Constant trajectories at ``f0`` [Hz] and ``level_db``, a flat spectrum unless given."""
        values = {
            "f0_mean": torch.tensor(hz_to_erb(f0), dtype=dtype),
            "f0_deviation": torch.zeros(timing.n_grid, dtype=dtype),
            "level_mean": torch.tensor(level_db, dtype=dtype),
            "level_deviation": torch.zeros(timing.n_grid, dtype=dtype),
            "spectrum_db": (
                torch.zeros(self.n_harmonics, dtype=dtype)
                if spectrum_db is None
                else torch.as_tensor(spectrum_db, dtype=dtype).clone()
            ),
        } | _timing_init(onset, duration, dtype)
        return {self.key(name): value for name, value in values.items()}

    def f0_erb(self, params: Params) -> torch.Tensor:
        """The f0 trajectory [ERB number] on the grid."""
        p = self.key
        return F0_TRAJECTORY.trajectory(params[p("f0_mean")], params[p("f0_deviation")])

    def render(self, params: Params, timing: SceneTiming) -> torch.Tensor:
        p = self.key
        f0 = erb_to_hz(self.f0_erb(params))
        level = HARMONIC_LEVEL_TRAJECTORY.trajectory(params[p("level_mean")], params[p("level_deviation")])
        spectrum = params[p("spectrum_db")]
        numbers = torch.arange(1, spectrum.shape[-1] + 1, dtype=f0.dtype)
        return _event(
            timing,
            numbers[:, None] * f0,
            level + spectrum[:, None],
            params[p("onset")],
            params[p("log_duration")].exp(),
        )

    def log_prior_terms(self, params: Params, timing: SceneTiming) -> list[torch.Tensor]:
        p = self.key
        return [
            F0_TRAJECTORY.log_prior(params[p("f0_mean")], params[p("f0_deviation")]),
            HARMONIC_LEVEL_TRAJECTORY.log_prior(params[p("level_mean")], params[p("level_deviation")]),
            # at the f0 the source actually has (its trajectory's mean, in ERB number),
            # not the mean of its prior
            log_prior_spectrum(params[p("spectrum_db")], erb_to_hz(self.f0_erb(params).mean())),
            log_prior_event_timing(params[p("onset")], params[p("log_duration")], timing.total_duration),
        ]

    def learning_rates(self) -> dict[str, float]:
        return {self.key(name): rate for name, rate in self.RATES.items()}


@dataclass(frozen=True)
class Whistle:
    """A whistle: one sinusoid whose frequency and level change over time."""

    name: str
    kind: ClassVar[str] = "whistle"
    RATES: ClassVar[dict[str, float]] = {
        "freq_mean": 5e-3,
        "freq_deviation": 2e-3,
        "level_mean": 0.5,
        "level_deviation": 0.05,
        "onset": 2e-4,
        "log_duration": 1e-3,
    }

    def key(self, parameter: str) -> str:
        return f"{self.name}.{parameter}"

    def initial(
        self,
        timing: SceneTiming,
        *,
        freq: float,
        level_db: float,
        onset: float,
        duration: float,
        dtype=torch.float64,
    ) -> Params:
        values = {
            "freq_mean": torch.tensor(hz_to_erb(freq), dtype=dtype),
            "freq_deviation": torch.zeros(timing.n_grid, dtype=dtype),
            "level_mean": torch.tensor(level_db, dtype=dtype),
            "level_deviation": torch.zeros(timing.n_grid, dtype=dtype),
        } | _timing_init(onset, duration, dtype)
        return {self.key(name): value for name, value in values.items()}

    def render(self, params: Params, timing: SceneTiming) -> torch.Tensor:
        p = self.key
        freq = erb_to_hz(F0_TRAJECTORY.trajectory(params[p("freq_mean")], params[p("freq_deviation")]))
        level = WHISTLE_LEVEL_TRAJECTORY.trajectory(params[p("level_mean")], params[p("level_deviation")])
        return _event(timing, freq[None], level[None], params[p("onset")], params[p("log_duration")].exp())

    def log_prior_terms(self, params: Params, timing: SceneTiming) -> list[torch.Tensor]:
        p = self.key
        return [
            F0_TRAJECTORY.log_prior(params[p("freq_mean")], params[p("freq_deviation")]),
            WHISTLE_LEVEL_TRAJECTORY.log_prior(params[p("level_mean")], params[p("level_deviation")]),
            log_prior_event_timing(params[p("onset")], params[p("log_duration")], timing.total_duration),
        ]

    def learning_rates(self) -> dict[str, float]:
        return {self.key(name): rate for name, rate in self.RATES.items()}


Source = Harmonic | Whistle


@dataclass(frozen=True)
class Scene:
    """A hypothesis: its sources, each with one event, in a sound of the given timing."""

    sources: tuple[Source, ...]
    timing: SceneTiming

    def __post_init__(self):
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")

    def render(self, params: Params) -> torch.Tensor:
        waveform = self.sources[0].render(params, self.timing)
        for source in self.sources[1:]:
            waveform = waveform + source.render(params, self.timing)
        return waveform

    def log_prior(self, params: Params) -> torch.Tensor:
        """Log prior of the sources' continuous parameters."""
        # summed term by term, left to right, so the rounding matches milestone (a)'s script
        terms = [term for source in self.sources for term in source.log_prior_terms(params, self.timing)]
        log_prior = terms[0]
        for term in terms[1:]:
            log_prior = log_prior + term
        return log_prior

    def structure_log_prior(self) -> float:
        """Log prior of the scene's discrete structure: number and types of sources."""
        return log_prior_structure([source.kind for source in self.sources], self.timing.total_duration)

    def learning_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for source in self.sources:
            rates |= source.learning_rates()
        return rates

    def log_joint(
        self, observed: torch.Tensor, cochleagram: Cochleagram, sigma: float = 10.0
    ) -> Callable[[Params], torch.Tensor]:
        """``params -> log p(observed | params) + log p(params)``, for the evidence.

        ``sigma`` is the likelihood's SD [dB] (BASS's value by default).
        """
        return lambda params: (
            gaussian_log_likelihood(observed, cochleagram(self.render(params)), sigma)
            + self.log_prior(params)
        )


@dataclass
class SceneResult:
    """A hypothesis's best fit, its log marginal likelihood and its structure prior."""

    fit: FitResult
    evidence: Evidence
    structure_log_prior: float
    fits: list[FitResult] = field(default_factory=list)
    variational: VariationalEvidence | None = None

    @property
    def log_posterior_laplace(self) -> float:
        """Unnormalized log posterior of the hypothesis, by the Laplace estimate."""
        return self.evidence.laplace + self.structure_log_prior

    @property
    def log_posterior_importance(self) -> float:
        """Unnormalized log posterior of the hypothesis, by importance sampling."""
        return self.evidence.importance + self.structure_log_prior

    @property
    def log_posterior_variational(self) -> float:
        """Unnormalized log posterior of the hypothesis, by importance sampling from the variational fit."""
        return self.variational.importance + self.structure_log_prior

    @property
    def log_posterior_elbo(self) -> float:
        """Unnormalized log posterior of the hypothesis, by the ELBO (a lower bound)."""
        return self.variational.elbo + self.structure_log_prior


def evaluate(
    scene: Scene,
    inits: list[Params],
    observed: torch.Tensor,
    cochleagram: Cochleagram,
    *,
    steps: int = 300,
    n_samples: int = 128,
    generator: torch.Generator | None = None,
    variational_steps: int = 0,
    sigma: float = 10.0,
) -> SceneResult:
    """Fit ``scene`` from each initialization, keep the best mode, and estimate its evidence.

    The Laplace and importance-sampling estimates are always made; with
    ``variational_steps``, so is the variational one, from the same mode.
    ``sigma`` is the likelihood's SD [dB] (BASS's value by default).
    """
    fits = [
        fit(
            scene.render,
            init,
            observed,
            cochleagram,
            learning_rates=scene.learning_rates(),
            steps=steps,
            sigma=sigma,
            log_prior=scene.log_prior,
        )
        for init in inits
    ]
    best = max(fits, key=lambda result: result.log_likelihood)
    evidence = log_evidence(
        scene.log_joint(observed, cochleagram, sigma), best.params, n_samples=n_samples, generator=generator
    )
    variational = None
    if variational_steps:
        variational = variational_evidence(
            scene.log_joint(observed, cochleagram, sigma),
            best.params,
            steps=variational_steps,
            generator=generator,
        )
    return SceneResult(best, evidence, scene.structure_log_prior(), fits, variational)
