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

The sigma and lengthscale of each Gaussian process are fixed at the
Table A.2 medians unless a source has ``infer_kernel=True`` (design C3 (B)).
Then each is a latent too, named e.g. ``"h0.f0_sigma_free"``: the value
mapped to the whole real line, under the paper's truncated normal prior
(see :class:`~sonore_inference.priors.KernelParameterPrior`).
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood
from sonore_inference.evidence import Evidence, VariationalEvidence, log_evidence, variational_evidence
from sonore_inference.fit import FitResult, fit
from sonore_inference.priors import (
    F0_KERNEL,
    F0_TRAJECTORY,
    GRID_STEP,
    HARMONIC_LEVEL_KERNEL,
    HARMONIC_LEVEL_TRAJECTORY,
    SPECTRUM_KERNEL,
    SPECTRUM_LENGTHSCALE_ERB,
    SPECTRUM_SIGMA_DB,
    WHISTLE_LEVEL_KERNEL,
    WHISTLE_LEVEL_TRAJECTORY,
    BoundedUniform,
    KernelParameterPrior,
    NormalGammaMarginal,
    log_duration_distribution,
    log_prior_structure,
    sample_structure,
    spectrum_distribution,
)
from sonore_inference.sources import _gate, trajectory_event

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


Sample = Callable[[str, torch.distributions.Distribution, int], torch.Tensor]
"""How a source's program meets its random variables: ``sample(name, distribution, term)``
returns the variable's value. Scoring returns the given parameter and adds
its log density to prior term ``term``; sampling draws from the
distribution; :meth:`Scene.model` calls ``pyro.sample`` (design C7)."""

# the prior terms of a source, summed in this order (milestone (a)'s order, which
# the exact reproduction of its results relies on)
TERM_FREQUENCY, TERM_LEVEL, TERM_SPECTRUM, TERM_TIMING = range(4)


def _as_list(value, n_events: int, what: str) -> list[float]:
    values = [value] * n_events if isinstance(value, (int, float)) else list(value)
    if len(values) != n_events:
        raise ValueError(f"{what}: expected {n_events} values, got {len(values)}")
    return values


def _timing_init(n_events: int, onset, duration, dtype) -> dict[str, torch.Tensor]:
    """Initial timing latents: the first onset and log duration, and with
    several events, every log duration and the log rests between events
    (``onset`` and ``duration`` then give each event's)."""
    if n_events == 1:
        return {
            "onset": torch.tensor(onset, dtype=dtype),
            "log_duration": torch.tensor(math.log(duration), dtype=dtype),
        }
    onsets = _as_list(onset, n_events, "onset")
    durations = _as_list(duration, n_events, "duration")
    rests = [onsets[j + 1] - onsets[j] - durations[j] for j in range(n_events - 1)]
    if min(rests) <= 0:
        raise ValueError("events of one source must not overlap")
    return {
        "onset": torch.tensor(onsets[0], dtype=dtype),
        "log_duration": torch.tensor(durations, dtype=dtype).log(),
        "log_rest": torch.tensor(rests, dtype=dtype).log(),
    }


def _per_event_deviation(timing: SceneTiming, values: list[float], onsets: list[float], dtype):
    """A constant mean and grid deviations that hold each event's value from its
    onset until the next event's (the first value before the first onset)."""
    mean = sum(values) / len(values)
    t = torch.arange(timing.n_grid, dtype=dtype) * GRID_STEP
    index = torch.zeros(timing.n_grid, dtype=torch.long)
    for j, onset in enumerate(onsets[1:], start=1):
        index[t >= onset] = j
    return torch.tensor(mean, dtype=dtype), torch.tensor(values, dtype=dtype)[index] - mean


class _Events:
    """The timing part of a source's program: sites, event times and grid memberships."""

    @staticmethod
    def sample(source, sample: Sample, timing: SceneTiming, dtype) -> tuple[list, list]:
        """Each event's onset and duration [s], from the source's timing sites [App. A.2, Eqns A.8-A.12]:
        the first onset uniform over the scene, then log durations and log rests
        whose shared (mu, lambda) are integrated out (design C2, C3 (D))."""
        p = source.key
        onset = sample(p("onset"), BoundedUniform(0.0, timing.total_duration, dtype), TERM_TIMING)
        if source.n_events == 1:
            log_duration = sample(p("log_duration"), log_duration_distribution(dtype), TERM_TIMING)
            return [onset], [log_duration.exp()]
        durations = sample(p("log_duration"), NormalGammaMarginal(source.n_events, dtype), TERM_TIMING).exp()
        rests = sample(p("log_rest"), NormalGammaMarginal(source.n_events - 1, dtype), TERM_TIMING).exp()
        onsets = [onset]
        for j in range(source.n_events - 1):
            onsets.append(onsets[-1] + durations[j] + rests[j])
        return onsets, list(durations)

    @staticmethod
    def memberships(source, timing: SceneTiming, onsets, durations, dtype) -> torch.Tensor | None:
        """How much each grid point lies inside each event (None with one event; design C4)."""
        if source.n_events == 1:
            return None
        t = torch.arange(timing.n_grid, dtype=dtype) * GRID_STEP
        events = zip(onsets, durations, strict=True)
        return torch.stack([_gate(t, onset, duration, timing.ramp) for onset, duration in events])


class _Scorer:
    """The ``sample`` of scoring: returns the given parameters and adds up their log densities by term."""

    def __init__(self, params: Params):
        self.params = params
        self.terms: dict[int, torch.Tensor] = {}

    def __call__(self, name: str, distribution, term: int) -> torch.Tensor:
        value = self.params[name]
        log_prob = distribution.log_prob(value)
        self.terms[term] = log_prob if term not in self.terms else self.terms[term] + log_prob
        return value

    def ordered_terms(self) -> list[torch.Tensor]:
        return [self.terms[term] for term in sorted(self.terms)]


class _Recorder:
    """A ``sample`` that draws from each distribution, or asks ``draw`` to, and keeps the values by name."""

    def __init__(self, draw=None):
        self.draw = draw or (lambda name, distribution: distribution.sample())
        self.params: Params = {}

    def __call__(self, name: str, distribution, term: int) -> torch.Tensor:
        self.params[name] = self.draw(name, distribution)
        return self.params[name]


def _for_pyro(distribution):
    """The same distribution as a Pyro one (Pyro needs its mixin; values are unchanged)."""
    from pyro.distributions.torch_distribution import TorchDistributionMixin

    pyro_distribution = copy.copy(distribution)
    pyro_distribution.__class__ = _pyro_class(type(distribution), TorchDistributionMixin)
    return pyro_distribution


@functools.cache
def _pyro_class(cls, mixin):
    return type(cls.__name__, (cls, mixin), {})


class _Source:
    """What harmonic sources and whistles share: names, events, timing and step sizes."""

    name: str
    n_events: int
    infer_kernel: bool
    RATES: ClassVar[dict[str, float]]
    # each Gaussian process: its fixed (sigma, lengthscale), the priors on them, and its prior term
    KERNELS: ClassVar[dict[str, tuple[tuple[float, float], tuple[KernelParameterPrior, ...], int]]]
    # step size of a kernel latent (its free coordinate)
    KERNEL_RATE: ClassVar[float] = 5e-2

    def key(self, parameter: str) -> str:
        return f"{self.name}.{parameter}"

    def _kernel_names(self) -> list[str]:
        if not self.infer_kernel:
            return []
        return [f"{gp}_{which}_free" for gp in self.KERNELS for which in ("sigma", "lengthscale")]

    def _parameter_names(self) -> list[str]:
        names = [name for name in self.RATES if name != "log_rest"]
        return names + (["log_rest"] if self.n_events > 1 else []) + self._kernel_names()

    def learning_rates(self) -> dict[str, float]:
        return {self.key(name): self.RATES.get(name, self.KERNEL_RATE) for name in self._parameter_names()}

    def _kernel_init(self, dtype) -> dict[str, torch.Tensor]:
        """The kernel latents at the fixed values, so a fit starts where one with them fixed does."""
        values = {}
        for name in self._kernel_names():
            gp, which, _ = name.rsplit("_", 2)
            index = 0 if which == "sigma" else 1
            fixed, priors = self.KERNELS[gp][0][index], self.KERNELS[gp][1][index]
            values[name] = torch.tensor(priors.free(fixed), dtype=dtype)
        return values

    def _kernel(self, sample: Sample, gp: str, dtype) -> tuple:
        """The GP's (sigma, lengthscale): fixed, or drawn through ``sample`` when inferred."""
        fixed, priors, term = self.KERNELS[gp]
        if not self.infer_kernel:
            return fixed
        return tuple(
            prior.value(sample(self.key(f"{gp}_{which}_free"), prior.distribution(dtype), term))
            for which, prior in zip(("sigma", "lengthscale"), priors, strict=True)
        )

    def _trajectory(self, sample: Sample, gp: str, timing: SceneTiming, memberships, dtype):
        """The GP trajectory's mean and deviations on the grid, through ``sample``."""
        prior, term = self.TRAJECTORIES[gp], self.KERNELS[gp][2]
        if self.infer_kernel:
            prior = prior.with_kernel(*self._kernel(sample, gp, dtype))
        mean = sample(self.key(f"{gp}_mean"), prior.mean_distribution(dtype), term)
        gaussian = prior.deviation_distribution(timing.n_grid, memberships, dtype)
        return mean, sample(self.key(f"{gp}_deviation"), gaussian, term)

    def event_times(self, params: Params, timing: SceneTiming) -> tuple[list, list]:
        """Each event's onset and duration [s]."""
        return _Events.sample(self, _Scorer(params), timing, params[self.key("onset")].dtype)

    def log_prior_terms(self, params: Params, timing: SceneTiming) -> list[torch.Tensor]:
        scorer = _Scorer(params)
        self.program(scorer, timing, params[self.key("onset")].dtype)
        return scorer.ordered_terms()

    def _render_events(self, params, timing, freqs, levels) -> torch.Tensor:
        onsets, durations = self.event_times(params, timing)
        waveform = _event(timing, freqs, levels, onsets[0], durations[0])
        for onset, duration in zip(onsets[1:], durations[1:], strict=True):
            waveform = waveform + _event(timing, freqs, levels, onset, duration)
        return waveform


@dataclass(frozen=True)
class Harmonic(_Source):
    """A harmonic source with ``n_harmonics`` harmonics of a time-varying f0, and ``n_events`` events."""

    name: str
    n_harmonics: int
    n_events: int = 1
    infer_kernel: bool = False
    kind: ClassVar[str] = "harmonic"
    TRAJECTORIES: ClassVar = {"f0": F0_TRAJECTORY, "level": HARMONIC_LEVEL_TRAJECTORY}
    KERNELS: ClassVar = {
        "f0": ((F0_TRAJECTORY.sigma, F0_TRAJECTORY.lengthscale), F0_KERNEL, TERM_FREQUENCY),
        "level": (
            (HARMONIC_LEVEL_TRAJECTORY.sigma, HARMONIC_LEVEL_TRAJECTORY.lengthscale),
            HARMONIC_LEVEL_KERNEL,
            TERM_LEVEL,
        ),
        "spectrum": ((SPECTRUM_SIGMA_DB, SPECTRUM_LENGTHSCALE_ERB), SPECTRUM_KERNEL, TERM_SPECTRUM),
    }
    # step sizes, in ERB number, dB, seconds and log seconds (those of milestone (a))
    RATES: ClassVar[dict[str, float]] = {
        "f0_mean": 5e-3,
        "f0_deviation": 2e-3,
        "level_mean": 0.5,
        "level_deviation": 0.1,
        "spectrum_db": 0.5,
        "onset": 2e-4,
        "log_duration": 1e-3,
        "log_rest": 1e-3,
    }

    def initial(
        self,
        timing: SceneTiming,
        *,
        f0,
        level_db: float,
        onset,
        duration,
        spectrum_db: torch.Tensor | None = None,
        dtype=torch.float64,
    ) -> Params:
        """Constant trajectories at ``f0`` [Hz] and ``level_db``, a flat spectrum unless given.

        With several events, ``onset`` and ``duration`` [s] give each event's,
        and ``f0`` may too (held from each onset to the next).
        """
        if self.n_events == 1 or isinstance(f0, (int, float)):
            f0_mean = torch.tensor(hz_to_erb(f0 if self.n_events == 1 else float(f0)), dtype=dtype)
            f0_deviation = torch.zeros(timing.n_grid, dtype=dtype)
        else:
            f0_mean, f0_deviation = _per_event_deviation(
                timing, [hz_to_erb(f) for f in _as_list(f0, self.n_events, "f0")], list(onset), dtype
            )
        values = (
            {
                "f0_mean": f0_mean,
                "f0_deviation": f0_deviation,
                "level_mean": torch.tensor(level_db, dtype=dtype),
                "level_deviation": torch.zeros(timing.n_grid, dtype=dtype),
                "spectrum_db": (
                    torch.zeros(self.n_harmonics, dtype=dtype)
                    if spectrum_db is None
                    else torch.as_tensor(spectrum_db, dtype=dtype).clone()
                ),
            }
            | _timing_init(self.n_events, onset, duration, dtype)
            | self._kernel_init(dtype)
        )
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
        return self._render_events(params, timing, numbers[:, None] * f0, level + spectrum[:, None])

    def program(self, sample: Sample, timing: SceneTiming, dtype=torch.float64) -> None:
        """The source's random variables in generative order, through ``sample``."""
        onsets, durations = _Events.sample(self, sample, timing, dtype)
        memberships = _Events.memberships(self, timing, onsets, durations, dtype)
        f0_mean, f0_deviation = self._trajectory(sample, "f0", timing, memberships, dtype)
        self._trajectory(sample, "level", timing, memberships, dtype)
        # at the f0 the source actually has (its trajectory's mean, in ERB number),
        # not the mean of its prior
        f0 = erb_to_hz(F0_TRAJECTORY.trajectory(f0_mean, f0_deviation).mean())
        spectrum = spectrum_distribution(self.n_harmonics, f0, *self._kernel(sample, "spectrum", dtype))
        sample(self.key("spectrum_db"), spectrum, TERM_SPECTRUM)


@dataclass(frozen=True)
class Whistle(_Source):
    """A whistle: one sinusoid whose frequency and level change over time, with ``n_events`` events."""

    name: str
    n_events: int = 1
    infer_kernel: bool = False
    kind: ClassVar[str] = "whistle"
    TRAJECTORIES: ClassVar = {"freq": F0_TRAJECTORY, "level": WHISTLE_LEVEL_TRAJECTORY}
    KERNELS: ClassVar = {
        "freq": ((F0_TRAJECTORY.sigma, F0_TRAJECTORY.lengthscale), F0_KERNEL, TERM_FREQUENCY),
        "level": (
            (WHISTLE_LEVEL_TRAJECTORY.sigma, WHISTLE_LEVEL_TRAJECTORY.lengthscale),
            WHISTLE_LEVEL_KERNEL,
            TERM_LEVEL,
        ),
    }
    RATES: ClassVar[dict[str, float]] = {
        "freq_mean": 5e-3,
        "freq_deviation": 2e-3,
        "level_mean": 0.5,
        "level_deviation": 0.05,
        "onset": 2e-4,
        "log_duration": 1e-3,
        "log_rest": 1e-3,
    }

    def initial(
        self,
        timing: SceneTiming,
        *,
        freq,
        level_db: float,
        onset,
        duration,
        dtype=torch.float64,
    ) -> Params:
        """Constant trajectories at ``freq`` [Hz] and ``level_db``; with several events,
        ``onset``, ``duration`` and ``freq`` may give each event's."""
        if self.n_events == 1 or isinstance(freq, (int, float)):
            freq_mean = torch.tensor(hz_to_erb(freq if self.n_events == 1 else float(freq)), dtype=dtype)
            freq_deviation = torch.zeros(timing.n_grid, dtype=dtype)
        else:
            freq_mean, freq_deviation = _per_event_deviation(
                timing, [hz_to_erb(f) for f in _as_list(freq, self.n_events, "freq")], list(onset), dtype
            )
        values = (
            {
                "freq_mean": freq_mean,
                "freq_deviation": freq_deviation,
                "level_mean": torch.tensor(level_db, dtype=dtype),
                "level_deviation": torch.zeros(timing.n_grid, dtype=dtype),
            }
            | _timing_init(self.n_events, onset, duration, dtype)
            | self._kernel_init(dtype)
        )
        return {self.key(name): value for name, value in values.items()}

    def render(self, params: Params, timing: SceneTiming) -> torch.Tensor:
        p = self.key
        freq = erb_to_hz(F0_TRAJECTORY.trajectory(params[p("freq_mean")], params[p("freq_deviation")]))
        level = WHISTLE_LEVEL_TRAJECTORY.trajectory(params[p("level_mean")], params[p("level_deviation")])
        return self._render_events(params, timing, freq[None], level[None])

    def program(self, sample: Sample, timing: SceneTiming, dtype=torch.float64) -> None:
        """The source's random variables in generative order, through ``sample``."""
        onsets, durations = _Events.sample(self, sample, timing, dtype)
        memberships = _Events.memberships(self, timing, onsets, durations, dtype)
        self._trajectory(sample, "freq", timing, memberships, dtype)
        self._trajectory(sample, "level", timing, memberships, dtype)


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
        """Log prior of the scene's discrete structure: number, types and events of sources."""
        return log_prior_structure(
            [source.kind for source in self.sources],
            self.timing.total_duration,
            [source.n_events for source in self.sources],
        )

    @classmethod
    def sample(
        cls, timing: SceneTiming, seed: int, *, n_harmonics: int = 10, infer_kernel: bool = False
    ) -> tuple[Scene, Params]:
        """A scene and its parameters drawn from the prior, from the same
        distributions that :meth:`log_prior` and :meth:`structure_log_prior` score.

        Noise sources are not implemented (design C5), so the structure is
        drawn from the prior conditioned on having none. Harmonic sources get
        ``n_harmonics`` harmonics, a simplification: the paper's have every
        harmonic below the Nyquist frequency. With ``infer_kernel`` the GPs'
        sigma and lengthscale are drawn too; otherwise they are the medians.
        """
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            structure = sample_structure(timing.total_duration, exclude=("noise",))
            sources = tuple(
                Harmonic(f"h{i}", n_harmonics, n_events, infer_kernel)
                if kind == "harmonic"
                else Whistle(f"w{i}", n_events, infer_kernel)
                for i, (kind, n_events) in enumerate(structure)
            )
            recorder = _Recorder()
            for source in sources:
                source.program(recorder, timing)
        return cls(sources, timing), recorder.params

    def model(
        self,
        observed: torch.Tensor | None = None,
        cochleagram: Cochleagram | None = None,
        sigma: float = 10.0,
    ) -> Params:
        """The scene as a Pyro model: every latent is a ``pyro.sample`` site named as
        in the parameters, with the distribution :meth:`log_prior` scores, and the
        likelihood of ``observed`` (a cochleagram) is a ``pyro.factor``.

        Needs Pyro (the optional ``pyro`` extra). The structure is fixed, so
        :meth:`structure_log_prior` is not a site; a trace's log density is
        :meth:`log_prior` (plus the likelihood), not the hypothesis's whole
        log joint (design C6, C7).
        """
        import pyro

        recorder = _Recorder(lambda name, distribution: pyro.sample(name, _for_pyro(distribution)))
        for source in self.sources:
            source.program(recorder, self.timing)
        if observed is not None:
            rendered = cochleagram(self.render(recorder.params))
            pyro.factor("likelihood", gaussian_log_likelihood(observed, rendered, sigma))
        return recorder.params

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
    report: Callable[[str], None] | None = None,
) -> SceneResult:
    """Fit ``scene`` from each initialization, keep the best mode, and estimate its evidence.

    The Laplace and importance-sampling estimates are always made; with
    ``variational_steps``, so is the variational one, from the same mode.
    ``sigma`` is the likelihood's SD [dB] (BASS's value by default).
    ``report``, if given, receives a message as each stage starts.
    """

    def fit_one(index, init):
        if report:
            report(f"fitting (Adam, {steps} steps) from start {index + 1} of {len(inits)}")
        return fit(
            scene.render,
            init,
            observed,
            cochleagram,
            learning_rates=scene.learning_rates(),
            steps=steps,
            sigma=sigma,
            log_prior=scene.log_prior,
        )

    fits = [fit_one(index, init) for index, init in enumerate(inits)]
    if report:
        report(f"Laplace and importance sampling ({n_samples} samples)")
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
            report=report,
        )
    return SceneResult(best, evidence, scene.structure_log_prior(), fits, variational)
