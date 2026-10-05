"""The renderers against the stimuli, and the fit on a case with a known answer."""

import numpy as np
import pytest
import torch

from sonore_inference.cochleagram import Cochleagram
from sonore_inference.fit import fit
from sonore_inference.sources import harmonic_tone, whistle, whistle_event
from sonore_inference.stimuli import mistuned_harmonic_stimulus

FS = 20_000


def render_stimulus_like(f0, number, percent):
    levels = torch.full((12,), 60.0, dtype=torch.float64)
    mistuning = torch.zeros(12, dtype=torch.float64)
    mistuning[number - 1] = f0 * percent / 100
    return harmonic_tone(
        torch.tensor(f0, dtype=torch.float64),
        levels,
        fs=FS,
        onset=0.05,
        duration=0.4,
        total_duration=0.5,
        ramp=0.01,
        mistuning_hz=mistuning,
    )


@pytest.mark.parametrize("number, percent", [(1, 0), (3, 20)])
def test_harmonic_tone_reproduces_the_stimulus(number, percent):
    ours = render_stimulus_like(200.0, number, percent).numpy()
    expected = mistuned_harmonic_stimulus(200.0, number, percent).data[:, 0]
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-12)


def test_whistle_is_one_harmonic():
    kwargs = dict(fs=FS, onset=0.01, duration=0.1, total_duration=0.2, ramp=0.01)
    freq, level = torch.tensor(440.0, dtype=torch.float64), torch.tensor(50.0, dtype=torch.float64)
    torch.testing.assert_close(
        whistle(freq, level, **kwargs), harmonic_tone(freq, level.reshape(1), **kwargs)
    )


def test_fit_recovers_f0_and_level():
    cochleagram = Cochleagram()
    kwargs = dict(fs=FS, onset=0.05, duration=0.2, total_duration=0.3, ramp=0.01)

    def render(params):
        return harmonic_tone(params["log_f0"].exp(), params["levels_db"], **kwargs)

    truth = {"log_f0": torch.tensor(np.log(200.0)), "levels_db": torch.full((6,), 60.0)}
    observed = cochleagram(render(truth)).detach()
    init = {"log_f0": torch.tensor(np.log(206.0)), "levels_db": torch.full((6,), 55.0)}
    result = fit(
        render, init, observed, cochleagram, learning_rates={"log_f0": 2e-3, "levels_db": 0.5}, steps=150
    )
    assert result.params["log_f0"].exp().item() == pytest.approx(200.0, abs=0.5)
    np.testing.assert_allclose(result.params["levels_db"].numpy(), 60.0, atol=1.0)
    assert result.log_likelihood > result.history[0]


def test_whistle_event_matches_whistle_at_whole_samples():
    freq, level = torch.tensor(640.0, dtype=torch.float64), torch.tensor(60.0, dtype=torch.float64)
    timing = dict(fs=20_000, total_duration=0.5, ramp=0.01)
    given = whistle(freq, level, onset=0.0731, duration=0.2, **timing)
    inferred = whistle_event(
        freq,
        level,
        torch.tensor(0.0731, dtype=torch.float64),
        torch.tensor(0.2, dtype=torch.float64),
        **timing,
    )
    np.testing.assert_allclose(inferred.numpy(), given.numpy(), atol=1e-12 * given.abs().max().item())


def test_whistle_event_timing_has_gradients():
    onset = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    duration = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    sound = whistle_event(
        torch.tensor(500.0, dtype=torch.float64),
        torch.tensor(60.0, dtype=torch.float64),
        onset,
        duration,
        fs=20_000,
        total_duration=0.5,
        ramp=0.01,
    )
    sound.pow(2).sum().backward()
    assert onset.grad.abs() > 0 and duration.grad.abs() > 0


def test_constant_trajectories_give_the_whistle_and_the_harmonic_tone():
    from sonore_inference.sources import trajectory_event

    kwargs = dict(fs=FS, total_duration=0.5, ramp=0.01)
    freq, level = torch.tensor(330.0, dtype=torch.float64), torch.tensor(55.0, dtype=torch.float64)
    onset, duration = torch.tensor(0.0512, dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64)
    constant = torch.ones(1, 51, dtype=torch.float64)
    ours = trajectory_event(freq * constant, level * constant, onset, duration, grid_step=0.01, **kwargs)
    expected = whistle_event(freq, level, onset, duration, **kwargs)
    np.testing.assert_allclose(ours.numpy(), expected.numpy(), rtol=0, atol=1e-12)
    numbers = torch.arange(1, 13, dtype=torch.float64)[:, None]
    harmonics = trajectory_event(
        200.0 * numbers * constant, 60.0 * constant.expand(12, 51), 0.05, 0.4, grid_step=0.01, **kwargs
    )
    np.testing.assert_allclose(
        harmonics.numpy(), render_stimulus_like(200.0, 1, 0).numpy(), rtol=0, atol=1e-11
    )


def test_a_frequency_glide_has_the_integrated_phase():
    from sonore_inference.sources import trajectory_event

    # a linear glide from 400 to 500 Hz over 0.5 s: phase 2 pi (400 t + 100 t^2)
    grid = torch.linspace(400.0, 500.0, 51, dtype=torch.float64)[None]
    level = torch.full((1, 51), 60.0, dtype=torch.float64)
    sound = trajectory_event(grid, level, 0.0, 0.5, grid_step=0.01, fs=FS, total_duration=0.5, ramp=1e-9)
    t = torch.arange(10_000, dtype=torch.float64) / FS
    expected = np.sqrt(2) * 1e-3 * torch.cos(2 * np.pi * (400 * t + 100 * t**2))
    # the phase is a left Riemann sum of the frequency: exact for a constant
    # frequency, off by a few hundredths of a radian here
    np.testing.assert_allclose(sound.numpy(), expected.numpy(), atol=0.02 * np.sqrt(2) * 1e-3)
