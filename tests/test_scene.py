"""Scenes: rendering, priors and step sizes add up over sources, and the evaluation keeps the best fit."""

import math

import numpy as np
import pytest
import torch

from sonore_inference.cochleagram import FFTCochleagram
from sonore_inference.priors import log_prior_structure
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle, erb_to_hz, evaluate, hz_to_erb
from sonore_inference.sources import whistle_event
from sonore_inference.stimuli import mistuned_harmonic_stimulus

FS = 20_000
TIMING = SceneTiming(fs=FS, total_duration=0.5, ramp=0.01)
EVENT = dict(onset=0.05, duration=0.4)


def test_erb_conversions_invert_each_other():
    for freq in [50.0, 200.0, 3000.0]:
        assert erb_to_hz(torch.tensor(hz_to_erb(freq), dtype=torch.float64)).item() == pytest.approx(freq)


def test_constant_harmonic_source_reproduces_the_stimulus():
    harmonic = Harmonic("h0", 12)
    params = harmonic.initial(TIMING, f0=200.0, level_db=60.0, **EVENT)
    ours = Scene((harmonic,), TIMING).render(params).numpy()
    expected = mistuned_harmonic_stimulus(200.0, 1, 0).data[:, 0]
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-9)


def test_a_scene_sums_its_sources():
    harmonic, whistle = Harmonic("h0", 6), Whistle("w0")
    params = harmonic.initial(TIMING, f0=150.0, level_db=55.0, **EVENT) | whistle.initial(
        TIMING, freq=470.0, level_db=50.0, onset=0.1, duration=0.2
    )
    scene = Scene((harmonic, whistle), TIMING)
    expected_whistle = whistle_event(
        torch.tensor(470.0, dtype=torch.float64),
        torch.tensor(50.0, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(0.2, dtype=torch.float64),
        fs=FS,
        total_duration=0.5,
        ramp=0.01,
    )
    np.testing.assert_allclose(
        (scene.render(params) - harmonic.render(params, TIMING)).numpy(),
        expected_whistle.numpy(),
        rtol=0,
        atol=1e-9,
    )
    assert scene.log_prior(params).item() == pytest.approx(
        harmonic.log_prior_terms(params, TIMING)[0].item()
        + sum(term.item() for term in harmonic.log_prior_terms(params, TIMING)[1:])
        + sum(term.item() for term in whistle.log_prior_terms(params, TIMING))
    )
    assert scene.structure_log_prior() == log_prior_structure(["harmonic", "whistle"], 0.5)


def test_step_sizes_cover_every_parameter_in_order():
    sources = (Harmonic("h0", 4), Harmonic("h1", 3), Whistle("w0"))
    params = {}
    for source in sources:
        kwargs = dict(f0=200.0) if isinstance(source, Harmonic) else dict(freq=500.0)
        params |= source.initial(TIMING, level_db=60.0, **EVENT, **kwargs)
    assert list(Scene(sources, TIMING).learning_rates()) == list(params)
    assert params["h1.spectrum_db"].shape == (3,)
    assert params["w0.freq_deviation"].shape == (TIMING.n_grid,)
    assert params["h0.log_duration"].item() == pytest.approx(math.log(0.4))


def test_source_names_must_be_unique():
    with pytest.raises(ValueError):
        Scene((Whistle("a"), Whistle("a")), TIMING)


def test_evaluate_keeps_the_best_initialization():
    timing = SceneTiming(fs=FS, total_duration=0.2, ramp=0.01)
    event = dict(onset=0.05, duration=0.1)
    cochleagram = FFTCochleagram()
    scene = Scene((Whistle("w0"),), timing)
    truth = Whistle("w0").initial(timing, freq=1000.0, level_db=60.0, **event)
    observed = cochleagram(scene.render(truth)).detach()
    far = Whistle("w0").initial(timing, freq=2000.0, level_db=60.0, **event)
    generator = torch.Generator().manual_seed(0)
    result = evaluate(scene, [far, truth], observed, cochleagram, steps=3, n_samples=4, generator=generator)
    assert len(result.fits) == 2
    assert result.fit is result.fits[1]
    assert math.isfinite(result.evidence.laplace)
    assert result.log_posterior_laplace == result.evidence.laplace + scene.structure_log_prior()
