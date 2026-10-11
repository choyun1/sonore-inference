"""Milestone (c) step 3: the GPs' sigma and lengthscale as latents (design C3 (B))."""

import math

import pytest
import torch

from sonore_inference.cochleagram import FFTCochleagram, gaussian_log_likelihood
from sonore_inference.priors import (
    F0_KERNEL,
    F0_TRAJECTORY,
    HARMONIC_LEVEL_KERNEL,
    SPECTRUM_KERNEL,
    WHISTLE_LEVEL_KERNEL,
    TruncatedNormal,
    inverse_softplus,
    softplus,
)
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle

TIMING = SceneTiming(fs=20_000, total_duration=0.5, ramp=0.01)
F64 = torch.float64

# Table A.2's quartiles (Q1, Q2, Q3) of each parameter
QUARTILES = [
    (F0_KERNEL[0], (3.0, 5.9, 9.0)),
    (F0_KERNEL[1], (None, 2.5, 5.5)),  # Q1 0.31 printed, 0.45 sampled (see priors.F0_KERNEL)
    (WHISTLE_LEVEL_KERNEL[0], (1.2, 1.3, 1.5)),
    (WHISTLE_LEVEL_KERNEL[1], (6.4, 6.9, 7.5)),
    (HARMONIC_LEVEL_KERNEL[0], (6.3, 7.8, 9.4)),
    (HARMONIC_LEVEL_KERNEL[1], (0.078, 0.18, 0.39)),
    (SPECTRUM_KERNEL[0], (8.8, 11.8, 14.9)),
    (SPECTRUM_KERNEL[1], (1.3, 4.7, 9.4)),
]


@pytest.mark.parametrize(("prior", "quartiles"), QUARTILES)
def test_kernel_priors_reproduce_the_quartiles_of_table_a2(prior, quartiles):
    torch.manual_seed(0)
    values = prior.value(prior.distribution().sample((100_000,)))
    assert values.min() >= prior.bounds[0] and values.max() <= prior.bounds[1]
    sampled = values.quantile(torch.tensor([0.25, 0.5, 0.75], dtype=F64))
    for got, printed in zip(sampled.tolist(), quartiles, strict=True):
        if printed is not None:
            # the paper's quartiles come from 5000 draws and are printed to two figures
            assert got == pytest.approx(printed, rel=0.12)


def test_truncated_normal_is_a_density_on_its_interval():
    d = TruncatedNormal(1.0, 2.0, -0.5, 4.0)
    grid = torch.linspace(-0.5, 4.0, 20_001, dtype=F64)
    assert torch.trapezoid(d.log_prob(grid).exp(), grid).item() == pytest.approx(1.0, abs=1e-8)
    assert d.log_prob(torch.tensor([-0.6, 4.1], dtype=F64)).isneginf().all()


def test_inverse_softplus():
    for y in (0.01, 0.18, 2.5, 33.0):
        assert softplus(torch.tensor(inverse_softplus(y), dtype=F64)).item() == pytest.approx(y, rel=1e-12)


def _scene(infer_kernel):
    harmonic = Harmonic("h0", 4, infer_kernel=infer_kernel)
    whistle = Whistle("w0", infer_kernel=infer_kernel)
    params = harmonic.initial(TIMING, f0=200.0, level_db=60.0, onset=0.05, duration=0.4) | whistle.initial(
        TIMING, freq=900.0, level_db=55.0, onset=0.1, duration=0.2
    )
    return Scene((harmonic, whistle), TIMING), params


def test_inferred_kernels_start_at_the_fixed_values():
    fixed, fixed_params = _scene(False)
    inferred, params = _scene(True)
    kernels = (("h0", ("f0", "level", "spectrum")), ("w0", ("freq", "level")))
    kernel_names = {
        f"{s}.{gp}_{which}_free" for s, gps in kernels for gp in gps for which in ("sigma", "lengthscale")
    }
    assert set(params) == set(fixed_params) | kernel_names
    assert set(inferred.learning_rates()) == set(params)
    torch.testing.assert_close(inferred.render(params), fixed.render(fixed_params))
    # the prior is the fixed one's plus the kernel latents' own prior
    kernel_prior = 0.0
    for source in inferred.sources:
        for gp, (_, priors, _) in source.KERNELS.items():
            for which, prior in zip(("sigma", "lengthscale"), priors, strict=True):
                kernel_prior += (
                    prior.distribution().log_prob(params[f"{source.name}.{gp}_{which}_free"]).item()
                )
    assert inferred.log_prior(params).item() == pytest.approx(
        fixed.log_prior(fixed_params).item() + kernel_prior, rel=1e-10
    )


def test_a_shorter_f0_lengthscale_makes_a_jump_more_probable():
    # the posterior of the lengthscale given a trajectory that jumps by 2 ERB at 250 ms
    jump = torch.where(torch.arange(TIMING.n_grid) < 25, 0.0, 2.0).to(F64)
    sigma_prior, lengthscale_prior = F0_KERNEL
    sigma = torch.tensor(5.9, dtype=F64)
    log_posteriors = []
    for lengthscale in (2.5, 0.05):
        raw = torch.tensor(inverse_softplus(lengthscale), dtype=F64)
        prior = F0_TRAJECTORY.with_kernel(sigma, softplus(raw))
        log_posteriors.append(
            prior.deviation_distribution(TIMING.n_grid).log_prob(jump).item()
            + lengthscale_prior.raw_distribution().log_prob(raw).item()
        )
    assert log_posteriors[1] > log_posteriors[0]


def test_kernel_latents_get_gradients():
    scene, params = _scene(True)
    params = {name: value.clone().requires_grad_() for name, value in params.items()}
    scene.log_prior(params).backward()
    for name, value in params.items():
        if name.endswith("_free"):
            assert torch.isfinite(value.grad) and value.grad != 0, name


def test_sampled_scenes_with_inferred_kernels_can_be_scored():
    for seed in range(6):
        scene, params = Scene.sample(TIMING, seed, n_harmonics=4, infer_kernel=True)
        assert set(params) == set(scene.learning_rates())
        assert any(name.endswith("_free") for name in params)
        assert math.isfinite(scene.log_prior(params).item())


def test_the_pyro_model_scores_inferred_kernels_like_the_scene():
    pyro = pytest.importorskip("pyro")
    scene, params = _scene(True)
    cochleagram = FFTCochleagram()
    observed = cochleagram(scene.render(params)).detach() + 1.0
    conditioned = pyro.poutine.condition(scene.model, data=params)
    trace = pyro.poutine.trace(conditioned).get_trace(observed, cochleagram)
    likelihood = gaussian_log_likelihood(observed, cochleagram(scene.render(params)), 10.0)
    expected = scene.log_prior(params) + likelihood
    assert trace.log_prob_sum().item() == pytest.approx(expected.item(), rel=1e-12)


def test_free_kernel_latents_have_a_proper_density_and_map_inside_the_bounds():
    for prior in (F0_KERNEL[1], SPECTRUM_KERNEL[0]):
        free = torch.linspace(-40, 40, 400_001, dtype=F64)
        density = prior.distribution().log_prob(free).exp()
        assert torch.trapezoid(density, free).item() == pytest.approx(1.0, abs=1e-6)
        values = prior.value(free)
        assert values.min() >= prior.bounds[0] * (1 - 1e-9) and values.max() <= prior.bounds[1] * (1 + 1e-9)
        assert prior.value(torch.tensor(prior.free(2.5), dtype=F64)).item() == pytest.approx(2.5, rel=1e-12)
