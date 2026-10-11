"""Milestone (c) step 2: several events per source, the prior's distributions, sampling and Pyro."""

import math
from collections import Counter

import numpy as np
import pytest
import torch

from sonore_inference.cochleagram import FFTCochleagram, gaussian_log_likelihood
from sonore_inference.priors import (
    F0_TRAJECTORY,
    NormalGammaMarginal,
    ZeroTruncatedPoisson,
    log_duration_distribution,
    log_prior_structure,
    sample_structure,
)
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle
from sonore_inference.sources import whistle_event

FS = 20_000
TIMING = SceneTiming(fs=FS, total_duration=0.5, ramp=0.01)
F64 = torch.float64


def test_normal_gamma_marginal_of_one_value_is_the_student_t():
    x = torch.linspace(-6, 4, 11, dtype=F64)
    torch.testing.assert_close(
        NormalGammaMarginal(1).log_prob(x[:, None]), log_duration_distribution().log_prob(x)
    )


def test_normal_gamma_marginal_integrates_to_one():
    grid = torch.linspace(-40, 38, 1561, dtype=F64)
    x1, x2 = torch.meshgrid(grid, grid, indexing="ij")
    density = NormalGammaMarginal(2).log_prob(torch.stack([x1, x2], -1)).exp()
    assert torch.trapezoid(torch.trapezoid(density, grid), grid).item() == pytest.approx(1.0, abs=2e-3)


def test_normal_gamma_marginal_matches_averaging_over_mu_and_tau():
    # the density at a point, against the Normal density averaged over draws of (mu, tau)
    torch.manual_seed(0)
    x = torch.tensor([-1.5, -0.7, -1.2], dtype=F64)
    draws = NormalGammaMarginal(1).sample((400_000,))  # only to exercise sample's shapes
    assert draws.shape == (400_000, 1)
    gamma = torch.distributions.Gamma(torch.tensor(2.5, dtype=F64), torch.tensor(1.0, dtype=F64))
    tau = gamma.sample((400_000,))
    mu = -1.0 + torch.randn(400_000, dtype=F64) / torch.sqrt(0.5 * tau)
    normal = torch.distributions.Normal(mu[:, None], 1 / torch.sqrt(tau)[:, None])
    log_normal = normal.log_prob(x).sum(-1)
    averaged = torch.logsumexp(log_normal, 0) - math.log(400_000)
    assert NormalGammaMarginal(3).log_prob(x).item() == pytest.approx(averaged.item(), abs=0.02)


def test_zero_truncated_poisson():
    counts = torch.arange(1, 80, dtype=F64)
    assert ZeroTruncatedPoisson(1.5).log_prob(counts).exp().sum().item() == pytest.approx(1.0)
    torch.manual_seed(0)
    assert (ZeroTruncatedPoisson(0.3).sample((10_000,)) >= 1).all()


def test_structure_prior_counts_events():
    # one stream of 12 tones against two streams of 8 and 4: the events' prior is 0.5^12 either way
    one = log_prior_structure(["whistle"], 3.1, [12])
    two = log_prior_structure(["whistle", "whistle"], 3.1, [8, 4])
    rate = 3.1
    assert two - one == pytest.approx(math.log(rate) - math.log(2) + math.log(2) - math.log(3))
    assert one == pytest.approx(log_prior_structure(["whistle"], 3.1) + 11 * math.log(0.5))


def test_sampled_structures_follow_the_structure_prior():
    # a multiset of (type, events) is drawn with probability exp(structure prior) / prod(count!):
    # the prior counts the n! labellings of a fitted mode (design C6), sampling only the distinct orders
    torch.manual_seed(0)
    draws = 60_000
    counts = Counter(tuple(sorted(sample_structure(1.0))) for _ in range(draws))
    structures = [
        (("whistle", 1),),
        (("harmonic", 2),),
        (("harmonic", 1), ("whistle", 1)),
        (("noise", 1),) * 2,
    ]
    for structure in structures:
        kinds = [kind for kind, _ in structure]
        events = [m for _, m in structure]
        repeats = math.prod(math.factorial(c) for c in Counter(structure).values())
        expected = math.exp(log_prior_structure(kinds, 1.0, events)) / repeats
        # within 4 standard errors; a wrong count of repeats would be off by a factor of 2
        standard_error = math.sqrt(expected * (1 - expected) / draws)
        assert counts[structure] / draws == pytest.approx(expected, abs=4 * standard_error)


def test_sampling_can_exclude_a_type():
    torch.manual_seed(1)
    for _ in range(200):
        assert all(kind != "noise" for kind, _ in sample_structure(2.0, exclude=("noise",)))


def test_one_membership_everywhere_is_the_one_event_kernel():
    n = 7
    soft = F0_TRAJECTORY.soft_cholesky(torch.ones(1, n, dtype=F64))
    torch.testing.assert_close(soft, F0_TRAJECTORY.cholesky(n))


ONSETS, DURATIONS, FREQS = [0.1, 0.25, 0.4], [0.1, 0.05, 0.08], [1000.0, 1260.0, 1000.0]


def test_several_events_render_as_separate_whistles():
    source = Whistle("w0", n_events=3)
    params = source.initial(TIMING, freq=FREQS, level_db=60.0, onset=ONSETS, duration=DURATIONS)
    assert params["w0.log_rest"].shape == (2,)
    expected = sum(
        whistle_event(
            torch.tensor(freq, dtype=F64),
            torch.tensor(60.0, dtype=F64),
            torch.tensor(onset, dtype=F64),
            torch.tensor(duration, dtype=F64),
            fs=FS,
            total_duration=0.5,
            ramp=0.01,
        )
        for freq, onset, duration in zip(FREQS, ONSETS, DURATIONS, strict=True)
    )
    np.testing.assert_allclose(source.render(params, TIMING).numpy(), expected.numpy(), rtol=0, atol=1e-9)
    assert list(Scene((source,), TIMING).learning_rates()) == list(params)


def test_events_must_not_overlap():
    with pytest.raises(ValueError):
        Whistle("w0", n_events=2).initial(TIMING, freq=1000.0, level_db=60.0, onset=[0.1, 0.15], duration=0.1)


def test_the_prior_of_several_events_is_smooth_in_their_timing():
    source = Harmonic("h0", 4, n_events=3)
    params = source.initial(TIMING, f0=[200.0, 300.0, 200.0], level_db=60.0, onset=ONSETS, duration=DURATIONS)
    params = {name: value.requires_grad_() for name, value in params.items()}
    log_prior = Scene((source,), TIMING).log_prior(params)
    assert math.isfinite(log_prior.item())
    log_prior.backward()
    for name in ["h0.onset", "h0.log_duration", "h0.log_rest"]:
        assert torch.isfinite(params[name].grad).all()
        assert params[name].grad.abs().sum() > 0


def test_sampled_scenes_can_be_scored_and_rendered():
    for seed in range(8):
        scene, params = Scene.sample(TIMING, seed, n_harmonics=4)
        assert all(source.kind != "noise" for source in scene.sources)
        assert set(params) == set(scene.learning_rates())
        assert math.isfinite(scene.log_prior(params).item())
        assert scene.render(params).shape == (round(0.5 * FS),)


def test_the_pyro_model_scores_like_the_scene():
    pyro = pytest.importorskip("pyro")
    harmonic, whistle = Harmonic("h0", 4), Whistle("w0", n_events=3)
    params = harmonic.initial(TIMING, f0=200.0, level_db=60.0, onset=0.05, duration=0.4) | whistle.initial(
        TIMING, freq=FREQS, level_db=55.0, onset=ONSETS, duration=DURATIONS
    )
    scene = Scene((harmonic, whistle), TIMING)
    cochleagram = FFTCochleagram()
    observed = cochleagram(scene.render(params)).detach() + 1.0
    conditioned = pyro.poutine.condition(scene.model, data=params)
    trace = pyro.poutine.trace(conditioned).get_trace(observed, cochleagram)
    sites = {name for name, node in trace.nodes.items() if node["type"] == "sample" and node["is_observed"]}
    assert sites == set(params) | {"likelihood"}
    likelihood = gaussian_log_likelihood(observed, cochleagram(scene.render(params)), 10.0)
    expected = scene.log_prior(params) + likelihood
    assert trace.log_prob_sum().item() == pytest.approx(expected.item(), rel=1e-12)
